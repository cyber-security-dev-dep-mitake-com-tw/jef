locals {
  # See variables.tf: the intra-op pool deliberately leaves two cores for the
  # event loop and the OS.
  jef_threads = var.jef_threads > 0 ? var.jef_threads : max(var.vcpus - 2, 1)

  use_static_ip = length(var.ip_addresses) > 0

  # Fail early and loudly rather than silently serving meaningless vectors.
  backbone_is_test = var.jef_backbone == "hashing"
}

resource "terraform_data" "guard_test_backbone" {
  lifecycle {
    precondition {
      condition     = !local.backbone_is_test
      error_message = "jef_backbone='hashing' is the deterministic test backbone with no semantics. Set a real model id before provisioning."
    }
    precondition {
      condition     = !local.use_static_ip || var.gateway != null
      error_message = "gateway is required when ip_addresses is set."
    }
    precondition {
      condition     = !local.use_static_ip || length(var.ip_addresses) >= var.node_count
      error_message = "ip_addresses must supply at least one address per node."
    }
  }
}

resource "proxmox_virtual_environment_vm" "jef" {
  count = var.node_count

  name        = "${var.vm_name_prefix}-${count.index + 1}"
  node_name   = var.proxmox_node
  description = "JEF System One decision engine (CPU-only)"
  tags        = var.tags

  depends_on = [terraform_data.guard_test_backbone]

  clone {
    vm_id = var.template_id
    full  = true
  }

  cpu {
    cores = var.vcpus
    # 'host' exposes AVX-512/AMX to the guest. On a CPU-only inference box this
    # is not a micro-optimisation -- it is most of the matmul throughput.
    type = "host"
  }

  memory {
    dedicated = var.memory_mb
  }

  disk {
    datastore_id = var.datastore_id
    interface    = "scsi0"
    size         = var.disk_gb
    discard      = "on"
    ssd          = true
  }

  network_device {
    bridge  = var.network_bridge
    vlan_id = var.vlan_id
  }

  agent {
    enabled = true
  }

  operating_system {
    type = "l26"
  }

  initialization {
    datastore_id = var.datastore_id

    dynamic "ip_config" {
      for_each = local.use_static_ip ? [1] : []
      content {
        ipv4 {
          address = var.ip_addresses[count.index]
          gateway = var.gateway
        }
      }
    }

    dynamic "ip_config" {
      for_each = local.use_static_ip ? [] : [1]
      content {
        ipv4 {
          address = "dhcp"
        }
      }
    }

    dns {
      servers = var.nameservers
    }

    user_account {
      username = "jef"
      keys     = var.ssh_public_keys
    }
  }

  lifecycle {
    ignore_changes = [
      # The guest agent reports a MAC the provider would otherwise churn on.
      network_device[0].mac_address,
    ]
  }
}

# Hand off to Ansible: provisioning ends here, configuration starts there.
resource "local_file" "ansible_inventory" {
  filename        = "${path.module}/../ansible/inventory/hosts.ini"
  file_permission = "0600"

  content = templatefile("${path.module}/templates/hosts.ini.tftpl", {
    hosts        = local.inventory_hosts
    jef_threads  = local.jef_threads
    jef_backbone = var.jef_backbone
    jef_image    = var.jef_image
    vcpus        = var.vcpus
  })
}

locals {
  inventory_hosts = [
    for i, vm in proxmox_virtual_environment_vm.jef : {
      name = vm.name
      # Prefer the cloud-init address; fall back to whatever the agent reports.
      address = local.use_static_ip ? split("/", var.ip_addresses[i])[0] : try(
        [for ip in flatten(vm.ipv4_addresses) : ip if ip != "127.0.0.1"][0],
        "UNRESOLVED"
      )
    }
  ]
}
