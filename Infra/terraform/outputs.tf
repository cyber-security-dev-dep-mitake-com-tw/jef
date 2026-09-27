output "vm_names" {
  description = "Provisioned VM names"
  value       = [for vm in proxmox_virtual_environment_vm.jef : vm.name]
}

output "addresses" {
  description = "Addresses Ansible will target"
  value       = [for h in local.inventory_hosts : h.address]
}

output "jef_threads" {
  description = "Computed intra-op thread count passed through to the guests"
  value       = local.jef_threads
}

output "inventory_path" {
  description = "Ansible inventory written by this apply"
  value       = local_file.ansible_inventory.filename
}

output "next_step" {
  value = "cd ../ansible && ansible-playbook -i inventory/hosts.ini site.yml"
}
