# Ubuntu CPU Deployment Guide

This guide is for a production-style deployment practice environment on an Ubuntu machine with:

- no dedicated public IP
- no bought domain
- no GPU
- local CPU OCR
- local or same-machine Ollama
- Nginx in front of the FastAPI app
- optional public exposure through Cloudflare Tunnel

This is the best current deployment practice path for this repo because the active application runtime is:

- local adaptive PaddleOCR
- local FastAPI app
- Ollama as the active classifier provider

If you want to automate this guide instead of running the steps by hand, use the Ansible assets in:

- `deployment/ansible/`
- `deployment/ansible/README.md`

The current Docker assets in this repo are GPU-oriented, so for a CPU-only Ubuntu machine the simpler and more realistic path is:

- Python virtual environment
- `systemd`
- Nginx
- optional Cloudflare Tunnel

## 1. Target Architecture

The deployment shape in this guide is:

```text
Internet
  |
  | HTTPS
  v
Cloudflare Tunnel (optional public access)
  |
  v
Nginx on Ubuntu (:80)
  |
  v
FastAPI app on 127.0.0.1:8000
  |
  +--> local PaddleOCR on CPU
  |
  +--> Ollama on 127.0.0.1:11434
```

If you only want LAN testing, skip the Cloudflare Tunnel section and use the Ubuntu machine's local IP.

## 2. Prerequisites

Recommended Ubuntu versions:

- Ubuntu 22.04 LTS
- Ubuntu 24.04 LTS

Install system packages:

```bash
sudo apt update
sudo apt install -y git curl nginx python3.11 python3.11-venv python3-pip apache2-utils
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

Create a deployment directory:

```bash
sudo mkdir -p /opt/doc-intel-engine
sudo chown -R $USER:$USER /opt/doc-intel-engine
cd /opt/doc-intel-engine
```

Clone the repo:

```bash
git clone <your-repo-url> .
```

## 3. Python Environment

Create the virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install app dependencies from the repo:

```bash
uv sync
```

Install the CPU Paddle runtime separately.

This repo intentionally does not pin Paddle runtime in `pyproject.toml`, because Paddle is environment-specific.

For CPU deployment:

```bash
uv pip install paddlepaddle
```

Verify Paddle:

```bash
uv run python -c "import paddle; print(paddle.__version__); print(paddle.device.is_compiled_with_cuda())"
```

Expected result on this machine:

```text
False
```

## 4. Install Ollama

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start it once for testing:

```bash
ollama serve
```

In another terminal, pull the current model used by this repo:

```bash
ollama pull qwen2.5:1.5b
```

Stop the foreground Ollama process after verifying it works. Later, `systemd` will manage it.

Quick health check:

```bash
curl http://127.0.0.1:11434/api/tags
```

## 5. Prepare Production Environment Variables

Create a production copy of the environment file:

```bash
cp .env .env.production
```

Edit it:

```bash
nano .env.production
```

Recommended CPU-only production-practice values:

```env
APP_ENV=production
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=INFO
GLOG_minloglevel=2

UPLOAD_DIR=/opt/doc-intel-engine/data/uploads
MAX_UPLOAD_SIZE_MB=25

ENABLE_API_KEY_AUTH=true
API_KEYS=replace-with-a-long-random-api-key
JWT_SECRET=replace-with-a-long-random-secret

OCR_DEVICE=cpu
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
OCR_GPU_DETECTION_MODEL=PP-OCRv5_server_det
OCR_GPU_RECOGNITION_MODEL=PP-OCRv4_server_rec_doc
OCR_CPU_DETECTION_MODEL=PP-OCRv4_mobile_det
OCR_CPU_RECOGNITION_MODEL=en_PP-OCRv4_mobile_rec

IMAGE_OCR_MAX_DIMENSION=720
IMAGE_OCR_JPEG_QUALITY=75

CLASSIFIER_PROVIDER=ollama
CLASSIFIER_MODEL=qwen2.5:1.5b
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_KEEP_ALIVE=5m

RATE_LIMIT=30/minute
CLASSIFICATION_TIMEOUT_SECONDS=30
CLASSIFICATION_MAX_PARALLEL_CHUNKS=3
CLASSIFICATION_FIRST_PAGE_TARGET_CHARS=700
CLASSIFICATION_FIRST_PAGE_MIN_CHARS=180
CLASSIFICATION_FIRST_PAGE_MAX_CHUNKS=6
CLASSIFICATION_FIRST_PAGE_BATCH_SIZE=3
CLASSIFICATION_EARLY_EXIT_CONFIDENCE=0.82
CLASSIFICATION_CHUNK_MAX_TOKENS=24
CLASSIFICATION_FINAL_MAX_TOKENS=16
OLLAMA_MAX_CONNECTIONS=10
TEXT_SNIPPET_LIMIT=5000
CLASSIFICATION_CHUNK_PAGES=2
```

You can also start from the committed template:

- [deployment/.env.production.example](/C:/Users/Imtiaz/Documents/GitHub/doc-classification/deployment/.env.production.example)

Why `OCR_DEVICE=cpu` here:

- this machine has no GPU
- forcing CPU avoids unnecessary GPU probing and makes startup behavior deterministic

## 6. Test the App Manually First

Make sure uploads directory exists:

```bash
mkdir -p /opt/doc-intel-engine/data/uploads
```

Start the backend manually:

```bash
set -a
source .env.production
set +a
source .venv/bin/activate
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Verify health:

```bash
curl http://127.0.0.1:8000/api/health
```

Verify the OCR path reports CPU in the health response.

Stop the server after manual validation.

## 7. Run Ollama as a Service

Check whether Ollama already installed a service:

```bash
systemctl status ollama
```

The Ollama installer normally creates the `ollama` service for you. After install, verify and enable it:

```bash
sudo systemctl enable ollama
sudo systemctl start ollama
sudo systemctl status ollama
```

Verify:

```bash
curl http://127.0.0.1:11434/api/tags
```

## 8. Run the Backend as a Service

Create a dedicated backend service:

```bash
sudo tee /etc/systemd/system/doc-intel.service > /dev/null <<'EOF'
[Unit]
Description=Doc Intel FastAPI Backend
After=network-online.target ollama.service
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/opt/doc-intel-engine
EnvironmentFile=/opt/doc-intel-engine/.env.production
ExecStart=/opt/doc-intel-engine/.venv/bin/uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

Change `User=ubuntu` if your Linux username is different.

A ready-to-use backend service template is included in the repo:

- [deployment/doc-intel.service](/C:/Users/Imtiaz/Documents/GitHub/doc-classification/deployment/doc-intel.service)

Start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable doc-intel
sudo systemctl start doc-intel
sudo systemctl status doc-intel
```

Useful logs:

```bash
journalctl -u doc-intel -f
journalctl -u ollama -f
```

## 9. Configure Nginx

Create an Nginx site:

```bash
sudo tee /etc/nginx/sites-available/doc-intel > /dev/null <<'EOF'
server {
    listen 80;
    server_name _;

    client_max_body_size 25M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 60;
        proxy_send_timeout 300;
        proxy_read_timeout 300;
    }
}
EOF
```

Enable the site:

```bash
sudo ln -sf /etc/nginx/sites-available/doc-intel /etc/nginx/sites-enabled/doc-intel
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

Verify locally:

```bash
curl http://127.0.0.1/api/health
```

If another machine on the same LAN can reach the Ubuntu machine, test:

```bash
curl http://<ubuntu-lan-ip>/api/health
```

A ready-to-use Nginx config is included in the repo:

- [deployment/nginx-doc-intel.conf](/C:/Users/Imtiaz/Documents/GitHub/doc-classification/deployment/nginx-doc-intel.conf)

## 10. Public Exposure Without Buying a Domain

Because this machine has no public IP and no bought domain, the easiest public access option is Cloudflare Quick Tunnel.

Important limitation from Cloudflare:

- Quick Tunnels are for testing and development
- not for production
- they currently have a hard limit of `200` in-flight requests

Install `cloudflared` on Ubuntu:

```bash
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
```

Expose the Nginx endpoint:

```bash
cloudflared tunnel --url http://127.0.0.1:80
```

Cloudflare will print a public `trycloudflare.com` URL. Anyone can access that URL from the internet.

Use this for:

- deployment practice
- outside-network demos
- basic end-to-end testing

Do not treat Quick Tunnel as final production hosting.

If you later want a more production-like Cloudflare Tunnel setup, you need a Cloudflare-managed domain and a remotely-managed tunnel.

## 11. Load Testing

This deployment is good enough to test:

- app-level rate limiting
- concurrent requests
- Nginx request handling
- OCR and classifier latency under load

### A. Simple health endpoint concurrency test

Install `ab` already came from `apache2-utils`.

Run:

```bash
ab -n 200 -c 20 http://127.0.0.1/api/health
```

This is useful for:

- checking Nginx + FastAPI path stability
- basic concurrency behavior

### B. Rate limit test

Current app rate limiting is controlled by:

```env
RATE_LIMIT=30/minute
```

Test repeated calls:

```bash
seq 1 50 | xargs -I{} -P10 curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1/api/health
```

You should eventually see `429` responses when limits are exceeded.

### C. Document classification concurrency test

For a sample file:

```bash
seq 1 10 | xargs -I{} -P3 curl -s -X POST \
  -H "X-API-Key: replace-with-a-long-random-api-key" \
  -F "file=@/opt/doc-intel-engine/tests/fixtures/sample.pdf" \
  http://127.0.0.1/api/v1/documents/classify
```

Adjust:

- `-P3` for concurrency
- the test file path
- API key value

Watch logs in another shell:

```bash
journalctl -u doc-intel -f
```

## 12. Persistence

The deployment should preserve at least:

- uploaded documents
- Paddle/PaddleX model cache

Current important paths:

- uploads:
  - `/opt/doc-intel-engine/data/uploads`
- PaddleX model cache:
  - `/home/<user>/.paddlex/official_models`

If you redeploy often, do not delete these casually. Preserving them avoids repeated downloads and slower cold starts.

## 13. Automatic Deployment Options

There are two good automation paths for this Ubuntu machine.

### Option A: GitHub Actions + self-hosted runner on Ubuntu

This is the best automation path if:

- the repo is private
- you want push-to-deploy behavior

Important GitHub warning:

- self-hosted runners should only be used with private repositories
- on public repos, pull requests from forks can run dangerous code on your runner machine

#### Step 1: Add a self-hosted runner

In GitHub:

- go to repository settings
- Actions
- Runners
- add a Linux self-hosted runner

GitHub will give you commands like:

```bash
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-<version>.tar.gz -L <github-runner-url>
tar xzf ./actions-runner-linux-x64-<version>.tar.gz
./config.sh --url https://github.com/<owner>/<repo> --token <token>
```

#### Step 2: Install the runner as a service

From GitHub's docs, on Linux with `systemd`:

```bash
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status
```

#### Step 3: Use a deployment workflow

Example workflow file:

```yaml
name: Deploy Ubuntu

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - name: Sync dependencies
        run: |
          source /opt/doc-intel-engine/.venv/bin/activate
          cd /opt/doc-intel-engine
          git pull --ff-only
          uv sync
          uv pip install paddlepaddle
      - name: Restart backend
        run: |
          sudo systemctl restart doc-intel
          sudo systemctl status doc-intel --no-pager
```

If you use this approach, keep the repository private.

### Option B: GitHub Actions + SSH deploy

This is safer than a self-hosted runner for many simple deployments.

Flow:

1. GitHub Actions runs in GitHub-hosted runners
2. the workflow SSHs into the Ubuntu machine
3. it runs:
   - `git pull`
   - `uv sync`
   - `systemctl restart doc-intel`

Typical deployment script on the server:

```bash
cd /opt/doc-intel-engine
git pull --ff-only
source .venv/bin/activate
uv sync
uv pip install paddlepaddle
sudo systemctl restart doc-intel
```

A reusable deploy script is included in the repo:

- [deployment/deploy_ubuntu.sh](/C:/Users/Imtiaz/Documents/GitHub/doc-classification/deployment/deploy_ubuntu.sh)

This is simpler operationally and avoids hosting a permanent GitHub runner on the machine.

## 14. Operational Checklist

After deployment, check:

```bash
curl http://127.0.0.1/api/health
systemctl status doc-intel
systemctl status ollama
sudo systemctl status nginx
```

If using Cloudflare Tunnel:

```bash
cloudflared tunnel --url http://127.0.0.1:80
```

And test the public URL from another network.

## 15. Recommended Practice Path

For your current environment, the cleanest progression is:

1. deploy on Ubuntu in CPU mode
2. run Nginx locally on that machine
3. verify `/api/health`
4. test rate limiting and concurrency on LAN
5. expose through Cloudflare Quick Tunnel for internet access
6. automate with GitHub Actions over SSH or a private self-hosted runner

## References

- Cloudflare Quick Tunnels:
  - [Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)
- Cloudflare remotely-managed tunnels:
  - [Create a tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-remote-tunnel/)
- GitHub self-hosted runners:
  - [Adding self-hosted runners](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)
  - [Configure runner as a service](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/configure-the-application)
