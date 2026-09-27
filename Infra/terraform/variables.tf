# --------------------------------------------------------------------------- #
# Proxmox connection
# --------------------------------------------------------------------------- #

variable "proxmox_endpoint" {
  description = "Proxmox VE API endpoint, e.g. https://pve.example.internal:8006/"
  type        = string
}

variable "proxmox_api_token" {
  description = "API token as user@realm!tokenid=uuid. Prefer PROXMOX_VE_API_TOKEN in the environment."
  type        = string
  sensitive   = true
}

variable "proxmox_node" {
  description = "Name of the Proxmox node to place VMs on"
  type        = string
}

variable "proxmox_insecure" {
  description = "Skip TLS verification. Only for labs with self-signed certs."
  type        = bool
  default     = false
}

variable "proxmox_ssh_username" {
  description = "SSH user on the Proxmox host, used by the provider for file uploads"
  type        = string
  default     = "root"
}

# --------------------------------------------------------------------------- #
# Cluster shape
# --------------------------------------------------------------------------- #

variable "node_count" {
  description = "Number of JEF server VMs"
  type        = number
  default     = 1
}

variable "vcpus" {
  description = <<-EOT
    vCPUs per VM.

    16 is the documented target. JEF has no GPU path at all -- the frozen
    backbone, head-only training and encoder-scale model exist precisely so that
    this number is the only compute knob that matters.
  EOT
  type        = number
  default     = 16

  validation {
    condition     = var.vcpus >= 4
    error_message = "JEF needs at least 4 vCPUs; below that the encoder dominates request latency."
  }
}

variable "memory_mb" {
  description = "RAM per VM. mmBERT-base in fp32 plus request headroom fits comfortably in 8 GiB; 16 GiB leaves room for the page cache."
  type        = number
  default     = 16384
}

variable "disk_gb" {
  description = "Root disk. Model weights land in HF_HOME and are the bulk of it."
  type        = number
  default     = 40
}

variable "jef_threads" {
  description = <<-EOT
    Backbone intra-op thread count (JEF_THREADS).

    Defaults to vcpus - 2 rather than vcpus: uvicorn's event loop and the OS
    need headroom, and oversubscribing the intra-op pool on a CPU-only box makes
    p99 latency worse, not better, because threads contend on the same caches.
  EOT
  type        = number
  default     = 0 # 0 -> computed as vcpus - 2 in locals
}

variable "jef_backbone" {
  description = "JEF_BACKBONE. 'hashing' is the deterministic test backbone and must never be used in production."
  type        = string
  default     = "jhu-clsp/mmBERT-base"
}

variable "jef_image" {
  description = "Container image to run"
  type        = string
  default     = "ghcr.io/cyber-security-dev-dep-mitake-com-tw/jef:latest"
}

# --------------------------------------------------------------------------- #
# Placement and networking
# --------------------------------------------------------------------------- #

variable "template_id" {
  description = "VM ID of a cloud-init enabled Debian 12 / Ubuntu 24.04 template to clone"
  type        = number
}

variable "datastore_id" {
  description = "Proxmox datastore for VM disks"
  type        = string
  default     = "local-lvm"
}

variable "network_bridge" {
  description = "Proxmox bridge to attach to"
  type        = string
  default     = "vmbr0"
}

variable "vlan_id" {
  description = "Optional VLAN tag. null leaves the interface untagged."
  type        = number
  default     = null
}

variable "ip_addresses" {
  description = "Static CIDR per VM, e.g. [\"10.0.10.51/24\"]. Empty list uses DHCP."
  type        = list(string)
  default     = []

  validation {
    condition     = length(var.ip_addresses) == 0 || alltrue([for ip in var.ip_addresses : can(cidrhost(ip, 0))])
    error_message = "ip_addresses entries must be CIDR notation, e.g. 10.0.10.51/24."
  }
}

variable "gateway" {
  description = "Default gateway, required when ip_addresses is set"
  type        = string
  default     = null
}

variable "nameservers" {
  description = "DNS servers for the guests"
  type        = list(string)
  default     = ["1.1.1.1", "9.9.9.9"]
}

variable "ssh_public_keys" {
  description = "Keys installed for the jef user via cloud-init"
  type        = list(string)
}

variable "vm_name_prefix" {
  type    = string
  default = "jef"
}

variable "tags" {
  type    = list(string)
  default = ["jef", "system-one", "cpu-only"]
}
