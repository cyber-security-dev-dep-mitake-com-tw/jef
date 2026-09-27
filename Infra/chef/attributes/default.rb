# JEF is CPU-only by design: frozen backbone, head-only training, encoder-scale
# model. Every default here is sized for that, and there is no GPU attribute.

default['jef']['image']   = 'ghcr.io/cyber-security-dev-dep-mitake-com-tw/jef:latest'
default['jef']['user']    = 'jef'
default['jef']['home']    = '/opt/jef'
default['jef']['port']    = 8080
default['jef']['cache_dir'] = '/var/lib/jef/huggingface'

# 'hashing' is the deterministic test stub and the recipe refuses to converge
# with it -- it produces well-formed answers with no meaning.
default['jef']['backbone'] = 'jhu-clsp/mmBERT-base'

# Leave two cores for the event loop and the OS. Oversubscribing the intra-op
# pool on a CPU-only box worsens p99 rather than improving throughput.
default['jef']['threads'] = [node['cpu']['total'].to_i - 2, 1].max

default['jef']['workers']        = 1
default['jef']['alpha']          = '0.10'
default['jef']['log_level']      = 'INFO'
default['jef']['max_questions']  = 256
default['jef']['memory_limit']   = '8g'

# Empty means the server falls back to the zero-shot head and an unfitted
# calibrator. It says so in /healthz and in its logs.
default['jef']['head_path']        = ''
default['jef']['calibration_path'] = ''
