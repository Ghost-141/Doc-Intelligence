Production-oriented document classification backend for `pdf`, `docx` and image uploads. The current pipeline uses direct text extraction when possible, adaptive local PaddleOCR for OCR, and a three-chunk Ollama vote to classify each document.

## Current Architecture

- `FastAPI` backend with versioned APIs
- Direct text extraction for digital PDFs, DOCX, markdown, and plain text
- One adaptive local `PaddleOCR` backend
  - GPU path when local Paddle CUDA is available
  - CPU fallback path otherwise
- Concurrent startup warmup for both OCR and classifier so the first request does not pay the full cold-start cost
- Ollama-based classification with:
  - persistent HTTP client
  - up to 3 representative chunks per document
  - concurrent chunk classification
  - majority-vote aggregation
  - heuristic fallback and override when the model returns a weak `other`
- Structured JSON logging, health reporting, and Prometheus metrics exposed at `/metrics`

## OCR Design

At backend startup, the OCR layer checks local Paddle CUDA support and picks one model profile:

### GPU path

```env
OCR_DEVICE=auto
OCR_GPU_DETECTION_MODEL=PP-OCRv5_server_det
OCR_GPU_RECOGNITION_MODEL=PP-OCRv4_server_rec_doc
```

### CPU fallback

```env
OCR_DEVICE=auto
OCR_CPU_DETECTION_MODEL=PP-OCRv4_mobile_det
OCR_CPU_RECOGNITION_MODEL=en_PP-OCRv4_mobile_rec
```

If you explicitly set `OCR_DEVICE=cpu`, the CPU path is forced even when CUDA is available.

## Runtime Flow

1. `POST /api/v1/documents/classify` accepts the upload.
2. `IngestionService` stores the file under `data/uploads`.
3. `ExtractionService` chooses direct-text extraction or OCR.

### PDF extraction flow

```
Probe page 1 for embedded text
  ├── text length >= MIN_DIRECT_TEXT_LENGTH
  │     └── extract remaining pages → return full digital text (no OCR)
  └── text length < MIN_DIRECT_TEXT_LENGTH
        └── fall through to OCR on page 1 only (PaddleOCR, scale=1.5)
```

4. Images are resized/compressed before OCR.
5. OCR runs through the selected PaddleOCR profile.
6. `ClassificationService` builds up to 3 vote chunks from extracted pages.
7. If there are many page chunks, it keeps representative chunks from the beginning, middle, and end.
8. If there are too few page chunks, it falls back to first-page paragraph chunks or splits the extracted text into 3 bounded chunks.
9. Ollama classifies the selected chunks concurrently.
10. The final class is chosen by majority vote.
11. If the votes split, confidence is used as the tie-break.
12. If the winning result is `other` but heuristics strongly match a known type such as `resume`, the heuristic result overrides it.

### Startup warmup flow

Both OCR and classifier warm up concurrently at startup:

```
[PaddleOCR init + warmup]       ← runs in thread executor
[Classifier init + warmup]      ← runs concurrently in event loop
         ↓
app ready — total time = max(ocr, classifier)
```

## Monitoring

The app exposes Prometheus metrics at:

- [http://localhost:8000/metrics](http://localhost:8000/metrics)

A separate monitoring stack is included under:

- [`deployment/monitoring/docker-compose.yml`](deployment/monitoring/docker-compose.yml)
- [`deployment/monitoring/README.md`](deployment/monitoring/README.md)

That stack runs:

- `Prometheus` for scraping and storing metrics
- `Grafana` for dashboards
- `blackbox-exporter` for uptime and health probes

### App metrics exposed by FastAPI

- `docintel_http_requests_total`
  - total HTTP requests by method, path, and status code
- `docintel_http_request_duration_seconds`
  - HTTP latency histogram by method and path
- `docintel_document_classifications_total`
  - completed classification requests by document type, predicted class, and success or failure
- `docintel_document_classification_duration_seconds`
  - end-to-end processing latency for document classification
- `docintel_document_stage_duration_seconds`
  - per-step timing for `ingestion`, `extraction`, and `classification`
- `docintel_document_upload_size_bytes`
  - uploaded file size distribution by extension
- `docintel_service_available`
  - whether OCR and classifier services are available after startup
- `docintel_startup_degraded`
  - whether the app started in degraded mode

### Probe metrics collected by Prometheus

- `probe_success{job="fastapi-alive"}`
  - whether the app is reachable
- `probe_success{job="fastapi-health"}`
  - whether `/api/health` returned an `ok` health response
- `probe_success{job="ollama-alive"}`
  - whether the Ollama server is reachable
- `probe_duration_seconds`
  - blackbox probe latency
- `probe_http_status_code`
  - HTTP status codes returned by the probed endpoints

### Grafana dashboards included

- `FastAPI Health`
  - app reachability, app health, Ollama reachability, probe latency, and status code
- `FastAPI App Metrics`
  - total classified docs
  - average processing time
  - average processing time of each step over time
  - documents classified per minute
  - average processing time by document type
  - request rate and latency
  - classification counts and failed classifications
  - upload size distribution
  - service availability and degraded startup state

## Repository Layout

```text
doc-intel-engine/
├── app/
│   ├── api/
│   ├── core/
│   ├── schemas/
│   ├── services/
│   └── main.py
├── data/
├── deployment/
│   └── docker-compose.yml
├── models/
├── scripts/
└── tests/
```

## Local Setup

Use Python `3.11`.

### 1. Create the environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

### 2. Install the Paddle runtime separately

Paddle is environment-specific and is intentionally not pinned inside `pyproject.toml`.

- CPU local setup: install `paddlepaddle`
- GPU local setup: install the matching `paddlepaddle-gpu` wheel for your CUDA stack

Example verification:

```bash
uv run python -c "import paddle; print(paddle.__version__); print(paddle.device.is_compiled_with_cuda())"
```

### 3. Start Ollama

```bash
ollama serve
ollama pull qwen2.5:1.5b
```

### 4. Start the backend

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

- [http://localhost:8000/](http://localhost:8000/)
- [http://localhost:8000/docs](http://localhost:8000/docs)
- [http://localhost:8000/api/health](http://localhost:8000/api/health)
- [http://localhost:8000/metrics](http://localhost:8000/metrics)

### Optional: start Prometheus and Grafana

If you want monitoring dashboards locally:

```bash
docker compose -f deployment/monitoring/docker-compose.yml up -d
```

Open:

- [http://localhost:9090](http://localhost:9090) for Prometheus
- [http://localhost:3001](http://localhost:3001) for Grafana

## GPU Setup Guide

Use this path when you want PaddleOCR to run on an NVIDIA GPU instead of falling back to CPU.

### 1. Host prerequisites

Before installing Python dependencies, confirm the machine is GPU-ready:

- NVIDIA driver is installed correctly
- `nvidia-smi` works on the host
- your Paddle GPU wheel matches the host CUDA stack
- if you plan to use Docker, NVIDIA Container Toolkit is installed

Quick host check:

```bash
nvidia-smi
```

If `nvidia-smi` fails, fix the driver/runtime first. The app cannot use GPU until the host GPU stack is healthy.

### 2. Install the project normally

Create and activate the virtual environment, then install the repo:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

### 3. Install a GPU Paddle runtime

Install a `paddlepaddle-gpu` build that matches your CUDA version. This repo does not pin one specific GPU wheel because the correct package depends on the machine.

Example pattern:

```bash
uv pip install paddlepaddle-gpu
```

If your environment needs a specific wheel index or version, use the Paddle install command that matches your CUDA runtime.

### 4. Verify Paddle can see CUDA

Run:

```bash
uv run python -c "import paddle; print(paddle.__version__); print(paddle.device.is_compiled_with_cuda())"
```

Expected result for a working GPU install:

```text
True
```

You can also confirm the project picks the GPU path by starting the API and checking `/api/health`.

### 5. Set GPU-oriented environment values

These values match the repo's active GPU OCR profile:

```env
OCR_DEVICE=auto
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
OCR_GPU_DETECTION_MODEL=PP-OCRv5_server_det
OCR_GPU_RECOGNITION_MODEL=PP-OCRv4_server_rec_doc
OCR_CPU_DETECTION_MODEL=PP-OCRv4_mobile_det
OCR_CPU_RECOGNITION_MODEL=en_PP-OCRv4_mobile_rec
```

Notes:

- `OCR_DEVICE=auto` lets the app select GPU when Paddle reports CUDA support
- `OCR_DEVICE=gpu:0` is reasonable if you want to express GPU intent explicitly
- if CUDA is unavailable at runtime, the app will fall back to CPU and `/api/health` will show that

### 6. Start the classifier backend

For the default local classifier path:

```bash
ollama serve
ollama pull qwen2.5:1.5b
```

### 7. Start the API and verify GPU usage

Start the backend:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then verify:

```bash
curl http://127.0.0.1:8000/api/health
```

In the OCR model section, look for GPU signals such as:

- `status: "loaded"`
- `uses_gpu: true`
- `details.active_device: "gpu:0"`
- `details.selected_path: "gpu"`

### 8. Docker GPU path

This repo already includes a GPU-oriented container setup:

- [Dockerfile](/C:/Users/Imtiaz/Documents/GitHub/Doc-Intelligence/Dockerfile)
- [deployment/docker-compose.yml](/C:/Users/Imtiaz/Documents/GitHub/Doc-Intelligence/deployment/docker-compose.yml)

Current GPU-relevant settings already present there:

- GPU-capable PaddleX base image in the Dockerfile
- `gpus: all`
- `ipc: host`
- `shm_size: 8gb`
- `OCR_DEVICE=auto`

Run the API container:

```bash
docker compose -f deployment/docker-compose.yml up --build api
```

For Docker GPU execution to work, the host must expose the NVIDIA runtime into containers. If the container starts but `/api/health` reports CPU, the usual issue is host-side GPU runtime configuration rather than app code.

### 9. Common failure modes

- `paddle.device.is_compiled_with_cuda()` returns `False`: wrong Paddle wheel or no CUDA-capable runtime
- `nvidia-smi` works but app still uses CPU: the installed Paddle package does not match the machine CUDA stack
- Docker sees no GPU: NVIDIA Container Toolkit or Docker GPU runtime is not configured
- `/api/health` reports `unavailable`: OCR startup failed; inspect startup logs for the exact Paddle/PaddleOCR error

## Environment Configuration

Current active `.env` controls:

- `APP_ENV`
- `HOST`
- `PORT`
- `LOG_LEVEL`
- `GLOG_minloglevel`
- `UPLOAD_DIR`
- `MAX_UPLOAD_SIZE_MB`
- `CATEGORIES`
- `ENABLE_API_KEY_AUTH`
- `API_KEYS`
- `JWT_SECRET`
- `RATE_LIMIT`
- `MIN_DIRECT_TEXT_LENGTH`
- `OCR_LANGUAGE`
- `OCR_TARGET_LANGUAGES`
- `OCR_DETECT_ORIENTATION`
- `OCR_DEVICE`
- `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK`
- `OCR_GPU_DETECTION_MODEL`
- `OCR_GPU_RECOGNITION_MODEL`
- `OCR_CPU_DETECTION_MODEL`
- `OCR_CPU_RECOGNITION_MODEL`
- `IMAGE_OCR_MAX_DIMENSION`
- `IMAGE_OCR_JPEG_QUALITY`
- `CLASSIFIER_PROVIDER`
- `CLASSIFIER_MODEL`
- `OLLAMA_BASE_URL`
- `OLLAMA_KEEP_ALIVE`
- `CLASSIFICATION_TIMEOUT_SECONDS`
- `CLASSIFICATION_MAX_PARALLEL_CHUNKS`
- `CLASSIFICATION_FIRST_PAGE_TARGET_CHARS`
- `CLASSIFICATION_FIRST_PAGE_MIN_CHARS`
- `CLASSIFICATION_FIRST_PAGE_MAX_CHUNKS`
- `CLASSIFICATION_FIRST_PAGE_BATCH_SIZE`
- `CLASSIFICATION_EARLY_EXIT_CONFIDENCE`
- `CLASSIFICATION_CHUNK_MAX_TOKENS`
- `CLASSIFICATION_FINAL_MAX_TOKENS`
- `OLLAMA_MAX_CONNECTIONS`
- `TEXT_SNIPPET_LIMIT`
- `CLASSIFICATION_CHUNK_PAGES`

## Docker Setup

The current Compose file is for:

- `api`

The API container expects Ollama to run on the host machine.

### Run the current stack

```bash
docker compose -f deployment/docker-compose.yml up --build
```

### Current Docker behavior

- API exposed on `http://localhost:8000`
- host Ollama is used through:
  - `http://host.docker.internal:11434`
- API container is configured with:
  - `OCR_DEVICE=auto`
  - `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`
- mounted volumes:
  - `../data -> /app/data`
  - `../models -> /app/models`

### Current Docker runtime notes

- The Dockerfile is using a Paddle GPU-capable base image
- Compose enables:
  - `gpus: all`
  - `ipc: host`
  - `shm_size: 8gb`
- In practice, Docker deployment still depends on the target machine having a compatible GPU runtime and matching Paddle environment

## Production Deployment (Ubuntu CPU)

The active production deployment runs without Docker:

- Python venv + `systemd` manages the FastAPI process
- Nginx sits in front on port 80 and handles rate limiting
- Ollama runs as a separate `systemd` service on `127.0.0.1:11434`
- ngrok provides optional public HTTPS exposure

Full step-by-step guide: [`deployment/UBUNTU_CPU_DEPLOYMENT.md`](deployment/UBUNTU_CPU_DEPLOYMENT.md)

Ansible automation for the same Ubuntu CPU path is available under:

- `deployment/ansible/site.yml`
- `deployment/ansible/README.md`

### systemd service

The backend service unit is at `deployment/doc-intel.service`:

```ini
[Service]
User=imtiaz
WorkingDirectory=/home/imtiaz/Documents/Doc-Intelligence/
EnvironmentFile=/home/imtiaz/Documents/Doc-Intelligence/.env.production
ExecStart=/home/imtiaz/Documents/Doc-Intelligence/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5
```

Manage it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable doc-intel
sudo systemctl restart doc-intel
journalctl -u doc-intel -f
```

### Nginx rate limiting

Rate limiting is handled entirely by Nginx — not by the app. The config is at `deployment/nginx-doc-intel.conf`:

```nginx
limit_req_zone $binary_remote_addr zone=doc_intel:10m rate=10r/m;

server {
    listen 80;
    server_name _;
    client_max_body_size 25M;

    location / {
        limit_req zone=doc_intel burst=5 nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:8000;
        ...
    }
}
```

Install and reload:

```bash
sudo cp deployment/nginx-doc-intel.conf /etc/nginx/sites-available/doc-intel
sudo ln -sf /etc/nginx/sites-available/doc-intel /etc/nginx/sites-enabled/doc-intel
sudo nginx -t && sudo systemctl reload nginx
```

### ngrok public exposure

ngrok tunnels the local Nginx port 80 to a public HTTPS URL:

```bash
ngrok http 80
```

ngrok prints a URL like `https://<id>.ngrok-free.app`. Use that as the base URL for external testing.

ngrok free tier limit: 40 req/min through the tunnel. Nginx rate limiting (10r/m, burst=5) fires before that.

### Deploy script

`deployment/deploy_ubuntu.sh` pulls, syncs deps, and restarts all three services:

```bash
bash deployment/deploy_ubuntu.sh
```

## Model Files

With the current local PaddleOCR setup, model files are cached by PaddleX under your user profile, for example:

```text
C:\Users\<username>\.paddlex\official_models\
```

Typical cached models for the current setup:

- `PP-OCRv5_server_det`
- `PP-OCRv4_server_rec_doc`
- `PP-OCRv4_mobile_det`
- `en_PP-OCRv4_mobile_rec`
- auxiliary orientation/unwarping models used by PaddleOCR

## API Endpoints

### Health

`GET /api/health`

Returns model status and active OCR/classifier configuration.

### Classify document

`POST /api/v1/documents/classify`

Current response shape:

- `doc_id`
- `filename`
- `doc_type`
- `classification`
- `confidence`
- `latency_ms`
- `ocr_text_preview`

If API-key auth is enabled, clients must send:

```http
X-API-Key: your-secret-key
```

## Warning Handling

The app currently suppresses only known startup noise:

- Paddle model hoster connectivity warnings
- `RequestsDependencyWarning`
- `No ccache found` startup warning
- low-signal GLOG warnings through `GLOG_minloglevel=2`

Real runtime errors, OCR failures, and application exceptions are still visible.

## Current Constraints

- OCR and classifier warm up concurrently at startup — total startup time is `max(ocr_warmup, classifier_warmup)` not the sum
- PDF OCR fallback only processes page 1 — sufficient for classification, not for full document extraction
- CPU OCR fallback is lighter but less accurate on stylized or noisy documents
- DB persistence is scaffolded but not wired into request handling
- document processing is still synchronous request-time work

## Tests

```bash
pytest
```

## Testing the Running Server

### Health check

```bash
curl http://127.0.0.1/api/health
```

### Classify a document

```bash
curl -X POST http://127.0.0.1/api/v1/documents/classify \
  -H "X-API-Key: <your-api-key>" \
  -F "file=@/path/to/sample.pdf"
```

Through ngrok:

```bash
curl -X POST https://<id>.ngrok-free.app/api/v1/documents/classify \
  -H "X-API-Key: <your-api-key>" \
  -F "file=@/path/to/sample.pdf"
```

### Rate limit test

Fire 15 requests with 5 in parallel — you should see a mix of `200` and `429`:

```bash
seq 1 15 | xargs -P5 -I{} curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1/api/health
```

A `429` from Nginx has no `X-Ngrok-Error-Code` header. A `429` from the ngrok tunnel limit does.

### Concurrency test

```bash
seq 1 10 | xargs -P3 -I{} curl -s -X POST \
  -H "X-API-Key: <your-api-key>" \
  -F "file=@tests/fixtures/sample.pdf" \
  http://127.0.0.1/api/v1/documents/classify
```

Watch logs live:

```bash
journalctl -u doc-intel -f
```
