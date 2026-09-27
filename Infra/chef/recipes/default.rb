#
# Cookbook:: jef
# Recipe:: default
#
# Converges a Debian/Ubuntu host into a JEF server.

if node['jef']['backbone'] == 'hashing'
  raise "node['jef']['backbone'] is 'hashing', the deterministic test backbone. " \
        'It returns well-formed answers with no semantics. Set a real model id.'
end

if node['cpu']['total'].to_i < 4
  raise "This host reports #{node['cpu']['total']} vCPUs. JEF needs at least 4; " \
        'the documented target is 16.'
end

if node['jef']['calibration_path'].empty?
  Chef::Log.warn(
    'No jef.calibration_path set: probabilities will be uncalibrated and ' \
    'confidence will reflect distribution concentration only. Do not gate ' \
    'automated SOAR actions on it.'
  )
end

apt_update 'periodic' do
  action :periodic
  frequency 3600
end

package %w(ca-certificates curl gnupg)

apt_repository 'docker' do
  uri "https://download.docker.com/linux/#{node['platform']}"
  components ['stable']
  arch node['kernel']['machine'] == 'x86_64' ? 'amd64' : 'arm64'
  key "https://download.docker.com/linux/#{node['platform']}/gpg"
  distribution node['lsb']['codename']
end

package %w(docker-ce docker-ce-cli containerd.io docker-compose-plugin)

service 'docker' do
  action %i(enable start)
end

group 'docker'

user node['jef']['user'] do
  system true
  shell '/usr/sbin/nologin'
  home node['jef']['home']
  manage_home true
end

group 'docker' do
  members [node['jef']['user']]
  append true
  action :modify
end

[node['jef']['home'], node['jef']['cache_dir']].each do |dir|
  directory dir do
    owner node['jef']['user']
    group node['jef']['user']
    mode '0750'
    recursive true
  end
end

template "#{node['jef']['home']}/jef.env" do
  source 'jef.env.erb'
  owner node['jef']['user']
  group node['jef']['user']
  mode '0640'
  notifies :restart, 'service[jef]', :delayed
end

template "#{node['jef']['home']}/docker-compose.yml" do
  source 'docker-compose.yml.erb'
  owner node['jef']['user']
  group node['jef']['user']
  mode '0640'
  notifies :restart, 'service[jef]', :delayed
end

template '/etc/systemd/system/jef.service' do
  source 'jef.service.erb'
  mode '0644'
  notifies :run, 'execute[systemctl daemon-reload]', :immediately
  notifies :restart, 'service[jef]', :delayed
end

execute 'systemctl daemon-reload' do
  action :nothing
end

service 'jef' do
  action %i(enable start)
end

# Converge is not done until the API actually answers -- and until it confirms
# it is not quietly serving the test backbone.
ruby_block 'verify jef is healthy' do
  block do
    require 'net/http'
    require 'json'
    require 'timeout'

    uri = URI("http://127.0.0.1:#{node['jef']['port']}/healthz")
    body = nil
    Timeout.timeout(600) do
      loop do
        begin
          resp = Net::HTTP.get_response(uri)
          if resp.code == '200'
            body = JSON.parse(resp.body)
            break
          end
        rescue StandardError # rubocop:disable Lint/SuppressedException
          # Still starting; a cold model load on CPU takes a while.
        end
        sleep 5
      end
    end

    raise 'Deployed server is serving the hashing test backbone.' if body['test_backbone']

    Chef::Log.info("JEF healthy — model=#{body['model']} calibrated=#{body['calibrated']}")
  end
end
