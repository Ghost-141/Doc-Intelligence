# Ansible Ubuntu Deployment

This playbook automates the current CPU-oriented Ubuntu deployment path that is otherwise described in `deployment/UBUNTU_CPU_DEPLOYMENT.md`.

## What it provisions

- Ubuntu packages for Python, Nginx, Git, and basic load testing
- `uv` for the deployment user
- repository checkout into `/opt/doc-intel-engine` by default
- Python virtual environment and locked dependency sync
- CPU Paddle runtime inside the virtual environment
- Ollama install, use of the installer-managed service, and model pull
- `.env.production` from Ansible variables
- `doc-intel` and `ollama` `systemd` services
- Nginx reverse proxy with rate limiting
- post-deploy health check against `http://127.0.0.1/api/health`

## Files to edit first

Update [`group_vars/all.yml`](/C:/Users/Imtiaz/Documents/GitHub/Doc-Intelligence/deployment/ansible/group_vars/all.yml):

- `doc_intel_repo_url`
- `doc_intel_env.API_KEYS`
- `doc_intel_env.JWT_SECRET`
- any host, path, model, or Nginx overrides

If your server login is not `ubuntu`, also update:

- `doc_intel_app_user`
- `doc_intel_app_group`
- `doc_intel_app_home`

## Inventory

Copy [`inventory.example.ini`](/C:/Users/Imtiaz/Documents/GitHub/Doc-Intelligence/deployment/ansible/inventory.example.ini) to `inventory.ini` and replace the placeholder host/IP.

## Run

```bash
ansible-playbook -i deployment/ansible/inventory.ini deployment/ansible/site.yml
```

If your SSH key is not the default:

```bash
ansible-playbook -i deployment/ansible/inventory.ini deployment/ansible/site.yml --private-key ~/.ssh/your-key.pem
```

## Notes

- The playbook assumes a CPU deployment and installs `paddlepaddle`.
- It keeps the existing non-Docker production shape: FastAPI + Ollama + `systemd` + Nginx.
- It assumes the Ollama installer creates the `ollama` service on the host.
- The checked-in Docker assets remain separate and GPU-oriented.
