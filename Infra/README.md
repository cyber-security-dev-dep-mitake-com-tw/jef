# Infra — deploying JEF on Proxmox

Five stacks, one target. They are not alternatives stacked for show: each covers
a different layer of the same deployment, and each is usable on its own if your
shop already standardises on one tool.

| Stack | Layer | Use it when |
|---|---|---|
| `terraform/` | **Provision** VMs on Proxmox VE (`bpg/proxmox`) | Default. Declarative, has a real state file. |
| `pulumi/` | **Provision**, same thing in Python | Your team prefers a real language over HCL. |
| `ansible/` | **Configure** the VMs Terraform/Pulumi created | Always — the provisioners hand off to this. |
| `chef/` | **Configure**, Chef Infra alternative | You already run a Chef Server. |
| `puppet/` | **Configure**, Puppet alternative | You already run a Puppet master. |

## The 16 vCPU constraint is load-bearing

JEF is CPU-only by design: frozen backbone, head-only training, encoder-scale
model. Every default here is sized for that and there is no GPU path anywhere in
this directory. `jef_threads` maps to the backbone's intra-op thread count,
which is the single biggest throughput lever on this target — see
`terraform/variables.tf` for why it defaults to `vcpus - 2` rather than `vcpus`.

## Quick start

```bash
# 1. Provision
cd terraform
cp terraform.tfvars.example terraform.tfvars   # fill in your Proxmox endpoint + token
terraform init && terraform apply

# 2. Configure (inventory is written by terraform)
cd ../ansible
ansible-playbook -i inventory/hosts.ini site.yml
```

## Secrets

Nothing in this directory should ever contain a credential. `*.tfvars`,
`Pulumi.<stack>.yaml` and the generated Ansible inventory are gitignored. Use
`PROXMOX_VE_API_TOKEN` in the environment, or Pulumi's encrypted config.
