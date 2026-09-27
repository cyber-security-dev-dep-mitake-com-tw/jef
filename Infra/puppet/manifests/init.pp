# @summary Deploys the JEF System One decision engine.
#
# JEF is CPU-only by design -- frozen backbone, head-only training,
# encoder-scale model -- so there is no GPU parameter anywhere in this class.
# The only compute knob that matters is $threads, and it deliberately defaults
# below the vCPU count.
#
# @param backbone
#   Model id for JEF_BACKBONE. 'hashing' is the deterministic test stub and is
#   rejected: it returns well-formed answers with no semantics.
# @param threads
#   Intra-op thread count. Defaults to vCPUs minus two, leaving headroom for the
#   event loop and the OS; oversubscribing this pool worsens p99 latency.
# @param calibration_path
#   Path to a fitted calibrator. Empty means probabilities are uncalibrated and
#   confidence reflects distribution concentration only.
class jef (
  String[1]           $image            = 'ghcr.io/cyber-security-dev-dep-mitake-com-tw/jef:latest',
  String[1]           $backbone         = 'jhu-clsp/mmBERT-base',
  String[1]           $user             = 'jef',
  Stdlib::Absolutepath $home            = '/opt/jef',
  Stdlib::Absolutepath $cache_dir       = '/var/lib/jef/huggingface',
  Stdlib::Port        $port             = 8080,
  Integer[1]          $threads          = max($facts['processors']['count'] - 2, 1),
  Integer[1]          $workers          = 1,
  String[1]           $alpha            = '0.10',
  String[1]           $log_level        = 'INFO',
  Integer[1]          $max_questions    = 256,
  String[1]           $memory_limit     = '8g',
  String               $head_path        = '',
  String               $calibration_path = '',
) {

  if $backbone == 'hashing' {
    fail("jef::backbone is 'hashing', the deterministic test backbone. It \
returns well-formed answers with no semantics. Set a real model id.")
  }

  if $facts['processors']['count'] < 4 {
    fail("This host reports ${facts['processors']['count']} vCPUs. JEF needs at \
least 4; the documented target is 16.")
  }

  if empty($calibration_path) {
    notify { 'jef-uncalibrated':
      message => 'No jef::calibration_path set: probabilities will be \
uncalibrated and confidence reflects distribution concentration only. Do not \
gate automated SOAR actions on it.',
    }
  }

  include apt

  apt::source { 'docker':
    location => "https://download.docker.com/linux/${downcase($facts['os']['name'])}",
    repos    => 'stable',
    release  => $facts['os']['distro']['codename'],
    key      => {
      'name'   => 'docker.asc',
      'source' => "https://download.docker.com/linux/${downcase($facts['os']['name'])}/gpg",
    },
  }

  package { ['docker-ce', 'docker-ce-cli', 'containerd.io', 'docker-compose-plugin']:
    ensure  => installed,
    require => Apt::Source['docker'],
  }

  service { 'docker':
    ensure  => running,
    enable  => true,
    require => Package['docker-ce'],
  }

  user { $user:
    ensure => present,
    system => true,
    home   => $home,
    shell  => '/usr/sbin/nologin',
    groups => ['docker'],
    require => Package['docker-ce'],
  }

  file { [$home, $cache_dir]:
    ensure => directory,
    owner  => $user,
    group  => $user,
    mode   => '0750',
  }

  file { "${home}/jef.env":
    ensure  => file,
    owner   => $user,
    group   => $user,
    mode    => '0640',
    content => epp('jef/jef.env.epp', {
      'backbone'         => $backbone,
      'port'             => $port,
      'workers'          => $workers,
      'alpha'            => $alpha,
      'log_level'        => $log_level,
      'max_questions'    => $max_questions,
      'threads'          => $threads,
      'head_path'        => $head_path,
      'calibration_path' => $calibration_path,
    }),
    notify  => Service['jef'],
  }

  file { "${home}/docker-compose.yml":
    ensure  => file,
    owner   => $user,
    group   => $user,
    mode    => '0640',
    content => epp('jef/docker-compose.yml.epp', {
      'image'        => $image,
      'home'         => $home,
      'port'         => $port,
      'cache_dir'    => $cache_dir,
      'cpus'         => $facts['processors']['count'],
      'memory_limit' => $memory_limit,
    }),
    notify  => Service['jef'],
  }

  file { '/etc/systemd/system/jef.service':
    ensure  => file,
    mode    => '0644',
    content => epp('jef/jef.service.epp', { 'home' => $home }),
    notify  => [Exec['jef-daemon-reload'], Service['jef']],
  }

  exec { 'jef-daemon-reload':
    command     => '/bin/systemctl daemon-reload',
    refreshonly => true,
  }

  service { 'jef':
    ensure  => running,
    enable  => true,
    require => [
      Service['docker'],
      File["${home}/docker-compose.yml"],
      File["${home}/jef.env"],
      Exec['jef-daemon-reload'],
    ],
  }

  # The run is not finished until the API answers and confirms it is not
  # serving the test backbone.
  exec { 'jef-health-gate':
    command   => "/usr/bin/curl -sf http://127.0.0.1:${port}/healthz | /usr/bin/grep -q '\"test_backbone\":false'",
    tries     => 60,
    try_sleep => 10,
    unless    => "/usr/bin/curl -sf http://127.0.0.1:${port}/healthz | /usr/bin/grep -q '\"test_backbone\":false'",
    require   => Service['jef'],
  }
}
