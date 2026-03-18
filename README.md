Production-oriented document classification backend for `pdf`, `docx` and image uploads. The current pipeline uses direct text extraction when possible, adaptive local PaddleOCR for OCR, and Ollama for document classification.

## Current Architecture

- `FastAPI` backend with versioned APIs
- Direct text extraction for digital PDFs, DOCX, markdown, and plain text
- One adaptive local `PaddleOCR` backend
  - GPU path when local Paddle CUDA is available
  - CPU fallback path otherwise
- Startup OCR warmup so the first request does not pay the full cold-start cost
- Ollama-based classification with:
  - persistent HTTP client
  - first-page paragraph probing
  - concurrent early-exit chunk classification
- Structured JSON logging and health reporting

## OCR Design

The app no longer uses PaddleOCR-VL.

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
4. Images are resized/compressed before OCR.
5. OCR runs through the selected PaddleOCR profile.
6. The classifier probes page 1 first using paragraph-like chunks.
7. Chunk classification runs concurrently in small batches.
8. If confidence is high enough, the classifier exits early.
9. Otherwise it falls back to broader document chunk classification.

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

- [http://localhost:8000/docs](http://localhost:8000/docs)
- [http://localhost:8000/api/health](http://localhost:8000/api/health)

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

- OCR startup is faster than before, but model warmup still happens at backend start
- CPU OCR fallback is lighter but less accurate on stylized or noisy documents
- DB persistence is scaffolded but not wired into request handling
- document processing is still synchronous request-time work

## Tests

```bash
pytest
```
