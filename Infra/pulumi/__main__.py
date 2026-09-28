"""Provision JEF VMs on Proxmox VE.

The Terraform stack in ``../terraform`` does the same job; this exists for teams
that would rather express infrastructure in Python than HCL. The two are kept
deliberately equivalent -- same defaults, same guardrails, same handoff to
Ansible -- so switching between them is a preference, not a migration.
"""

from __future__ import annotations

import pulumi
import pulumi_proxmoxve as proxmox

config = pulumi.Config()
stack = pulumi.get_stack()

node_count = config.get_int("nodeCount") or 1
vcpus = config.get_int("vcpus") or 16
memory_mb = config.get_int("memoryMb") or 16384
disk_gb = config.get_int("diskGb") or 40
backbone = config.get("backbone") or "jhu-clsp/mmBERT-base"
proxmox_node = config.require("proxmoxNode")
template_id = config.require_int("templateId")
datastore_id = config.get("datastoreId") or "local-lvm"
network_bridge = config.get("networkBridge") or "vmbr0"
ssh_keys = config.require_object("sshPublicKeys")
ip_addresses: list[str] = config.get_object("ipAddresses") or []
gateway = config.get("gateway")

# --------------------------------------------------------------------------- #
# Guardrails -- fail at plan time, not after the VM is already running.
# --------------------------------------------------------------------------- #

if backbone == "hashing":
    raise pulumi.RunError(
        "backbone='hashing' is the deterministic test stub: it returns "
        "well-formed answers with no semantics. Set a real model id."
    )
if vcpus < 4:
    raise pulumi.RunError(
        f"vcpus={vcpus}: JEF needs at least 4. The documented target is 16, and "
        "there is no GPU path -- CPU is the only compute knob."
    )
if ip_addresses and not gateway:
    raise pulumi.RunError("gateway is required when ipAddresses is set")
if ip_addresses and len(ip_addresses) < node_count:
    raise pulumi.RunError("ipAddresses must supply one address per node")

# Leave two cores for the event loop and the OS; oversubscribing the intra-op
# pool on a CPU-only box degrades p99 rather than improving throughput.
jef_threads = max(vcpus - 2, 1)

vms = []
for i in range(node_count):
    static_ip = ip_addresses[i] if ip_addresses else None

    vm = proxmox.vm.VirtualMachine(
        f"jef-{i + 1}",
        name=f"jef-{i + 1}",
        node_name=proxmox_node,
        description="JEF System One decision engine (CPU-only)",
        tags=["jef", "system-one", "cpu-only", stack],
        clone=proxmox.vm.VirtualMachineCloneArgs(vm_id=template_id, full=True),
        cpu=proxmox.vm.VirtualMachineCpuArgs(
            cores=vcpus,
            # 'host' exposes AVX-512/AMX to the guest, which is most of the
            # matmul throughput on a CPU-only inference box.
            type="host",
        ),
        memory=proxmox.vm.VirtualMachineMemoryArgs(dedicated=memory_mb),
        disks=[
            proxmox.vm.VirtualMachineDiskArgs(
                datastore_id=datastore_id,
                interface="scsi0",
                size=disk_gb,
                discard="on",
                ssd=True,
            )
        ],
        network_devices=[proxmox.vm.VirtualMachineNetworkDeviceArgs(bridge=network_bridge)],
        agent=proxmox.vm.VirtualMachineAgentArgs(enabled=True),
        operating_system=proxmox.vm.VirtualMachineOperatingSystemArgs(type="l26"),
        initialization=proxmox.vm.VirtualMachineInitializationArgs(
            datastore_id=datastore_id,
            ip_configs=[
                proxmox.vm.VirtualMachineInitializationIpConfigArgs(
                    ipv4=proxmox.vm.VirtualMachineInitializationIpConfigIpv4Args(
                        address=static_ip or "dhcp",
                        gateway=gateway if static_ip else None,
                    )
                )
            ],
            user_account=proxmox.vm.VirtualMachineInitializationUserAccountArgs(
                username="jef", keys=list(ssh_keys)
            ),
        ),
    )
    vms.append(vm)


def _address(index: int, vm: proxmox.vm.VirtualMachine) -> pulumi.Output[str]:
    if ip_addresses:
        return pulumi.Output.from_input(ip_addresses[index].split("/")[0])
    return vm.ipv4_addresses.apply(
        lambda addrs: next(
            (ip for group in (addrs or []) for ip in group if ip != "127.0.0.1"),
            "UNRESOLVED",
        )
    )


addresses = pulumi.Output.all(*[_address(i, vm) for i, vm in enumerate(vms)])

pulumi.export("vm_names", [vm.name for vm in vms])
pulumi.export("addresses", addresses)
pulumi.export("jef_threads", jef_threads)
pulumi.export(
    "ansible_inventory",
    addresses.apply(
        lambda addrs: "\n".join(
            ["[jef]"]
            + [f"jef-{i + 1} ansible_host={a}" for i, a in enumerate(addrs)]
            + [
                "",
                "[jef:vars]",
                "ansible_user=jef",
                f"jef_backbone={backbone}",
                f"jef_threads={jef_threads}",
            ]
        )
    ),
)
pulumi.export("next_step", "pulumi stack output ansible_inventory > ../ansible/inventory/hosts.ini")
