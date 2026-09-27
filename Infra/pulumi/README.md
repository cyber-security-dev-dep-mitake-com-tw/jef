# Pulumi stack

Equivalent to `../terraform`, in Python. Pick one; running both against the same
Proxmox node will fight over VM IDs.

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
pulumi stack init dev
pulumi config set jef:proxmoxNode pve01
pulumi config set jef:templateId 9000
pulumi config set --path jef:sshPublicKeys[0] "ssh-ed25519 AAAA..."
pulumi config set --secret proxmoxve:apiToken "terraform@pve!jef=..."
pulumi up

pulumi stack output ansible_inventory > ../ansible/inventory/hosts.ini
```

Guardrails match the Terraform stack: it refuses to plan with the `hashing` test
backbone or fewer than 4 vCPUs.
