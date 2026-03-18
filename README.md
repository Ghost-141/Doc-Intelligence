# doc-intel-engine

Production-oriented multilingual document classification backend for `pdf`, `docx`, `markdown`, `txt`, and image uploads. The current system combines direct text extraction, an adaptive PaddleOCR path that selects GPU or CPU models at startup, and Ollama-based document classification.

## Current Status

- `FastAPI` backend with versioned APIs and structured JSON logging
- Direct text extraction for digital PDFs, DOCX, markdown, and plain text
- Adaptive `PaddleOCR` startup path with GPU-first and CPU-fallback model selection
- Bangla/English-aware downstream classification flow
- First-page paragraph probing with concurrent early-exit classification
- Persistent classifier HTTP client with optional Ollama keep-alive
- Dockerized API + model-serving stack

## Repository Layout

```text
doc-intel-engine/
├── app/
│   ├── api/
│   ├── core/
│   ├── models/
│   ├── schemas/
│   ├── services/
│   └── main.py
├── deployment/
│   └── docker-compose.yml
├── models/
├── scripts/
└── tests/
```

## Runtime Flow

1. `POST /api/v1/documents/classify` accepts a file upload.
2. `IngestionService` validates the file and persists it under `data/uploads`.
3. `ExtractionService` picks direct-text extraction or OCR.
4. Images are resized/compressed before OCR.
5. On startup, the OCR service selects a GPU-heavy PaddleOCR profile if CUDA is healthy; otherwise it loads a lighter CPU profile.
6. The classifier first probes page 1 by merging OCR output into meaningful paragraph chunks.
7. Those paragraph chunks are classified concurrently in small batches.
8. If confidence is strong enough, classification stops early.
9. Otherwise the system falls back to broader document chunk classification.

## Local Setup

Use Python `3.11` or `3.12`.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
uvicorn app.main:app
```

Install the Paddle runtime separately for your environment before starting the backend:

- CPU-only local setup: install the CPU Paddle wheel.
- GPU local setup: install the GPU Paddle wheel that matches your CUDA and cuDNN stack.

Open [http://localhost:8000/docs](http://localhost:8000/docs).

## Required Runtime Pieces

- `paddlepaddle` installed in the active environment
- Ollama

## Docker Setup

The current compose stack matches the current repo and includes:

- `api`
- optional `vllm`

### Run current stack with host Ollama

```bash
docker compose -f deployment/docker-compose.yml up --build
```

This brings up the API on `http://localhost:8000` and points it to Ollama running on the host at `http://host.docker.internal:11434`.

### Docker Notes

- The API image now uses the official PaddleX GPU runtime base image for CUDA `12.6` and cuDNN `9.5`.
- The API container runs with:
  - `gpus: all`
  - `ipc: host`
  - `shm_size: 8gb`
- The API container mounts:
  - `../data` to `/app/data`
  - `../models` to `/app/models`
- The API container overrides runtime URLs so internal service discovery works:
  - `OLLAMA_BASE_URL=http://host.docker.internal:11434`
  - `OCR_DEVICE=auto`

## Classifier Backends

### Ollama

Recommended for the current active setup:

```env
CLASSIFIER_PROVIDER=ollama
CLASSIFIER_MODEL=qwen2.5:1.5b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_KEEP_ALIVE=5m
```

## PaddleOCR Setup

The OCR service no longer uses PaddleOCR-VL. It loads one of two PaddleOCR profiles when the backend starts.

### GPU Path

Used automatically when local Paddle CUDA support is available and `OCR_DEVICE` is not forced to `cpu`.

```env
OCR_DEVICE=auto
OCR_GPU_DETECTION_MODEL=PP-OCRv5_server_det
OCR_GPU_RECOGNITION_MODEL=PP-OCRv4_server_rec_doc
```

This path is intended for:

- scanned PDFs
- denser layouts
- higher-accuracy OCR when a local GPU is healthy

### CPU Fallback Path

Used automatically when CUDA is unavailable, unhealthy, or `OCR_DEVICE=cpu`.

```env
OCR_CPU_DETECTION_MODEL=PP-OCRv4_mobile_det
OCR_CPU_RECOGNITION_MODEL=en_PP-OCRv4_mobile_rec
```

This path is intended for:

- CPU-only environments
- Docker environments without CUDA-enabled Paddle
- lower-memory fallback operation

## PaddleOCR Model Guidance For Inference Speed

### Fastest

The active OCR model pair is now split by device.

### Faster CPU Fallback

Use lighter CPU models when latency matters most:

```env
OCR_CPU_DETECTION_MODEL=PP-OCRv4_mobile_det
OCR_CPU_RECOGNITION_MODEL=en_PP-OCRv4_mobile_rec
```

Best for:

- CPU-only deployment
- simple English-heavy images
- lower-latency OCR

Tradeoff:

- weaker accuracy on stylized text, dense layouts, and noisy scans

### Stronger GPU Path

Use heavier GPU models when a local CUDA runtime is healthy:

```env
OCR_GPU_DETECTION_MODEL=PP-OCRv5_server_det
OCR_GPU_RECOGNITION_MODEL=PP-OCRv4_server_rec_doc
```

Best for:

- scanned PDFs
- formal documents
- OCR quality over speed in GPU-capable environments

Tradeoff:

- higher memory use than the CPU fallback path

### Image Compression Tuning

These settings also affect OCR speed and quality:

```env
IMAGE_OCR_MAX_DIMENSION=720
IMAGE_OCR_JPEG_QUALITY=75
```

Lower values improve speed but may hurt OCR quality. Increase them for difficult documents.

## Current `.env` Controls

- `OCR_DEVICE`
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
- `CLASSIFICATION_CHUNK_PAGES`
- `CLASSIFICATION_MAX_PARALLEL_CHUNKS`
- `CLASSIFICATION_FIRST_PAGE_TARGET_CHARS`
- `CLASSIFICATION_FIRST_PAGE_MIN_CHARS`
- `CLASSIFICATION_FIRST_PAGE_MAX_CHUNKS`
- `CLASSIFICATION_FIRST_PAGE_BATCH_SIZE`
- `CLASSIFICATION_EARLY_EXIT_CONFIDENCE`
- `CLASSIFICATION_CHUNK_MAX_TOKENS`
- `CLASSIFICATION_FINAL_MAX_TOKENS`
- `TEXT_SNIPPET_LIMIT`
- `ENABLE_API_KEY_AUTH`
- `API_KEYS`

## API Notes

- `GET /api/health` returns model status and GPU usage signals
- `POST /api/v1/documents/classify` returns:
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

## Current Constraints

- the OCR service auto-selects a GPU or CPU model pair at startup based on local Paddle CUDA support
- the CPU fallback models are faster and lighter but can hurt stylized-text accuracy
- DB persistence is scaffolded but not wired into request handling yet
- background job processing is not implemented yet

## Tests

```bash
pytest
```

## Recommended Next Steps

1. Add async job processing for large PDFs and batch jobs.
2. Persist OCR/classification results by document hash.
3. Add classifier fallback when a small model returns invalid structured output.
4. Benchmark mobile versus server PaddleOCR fallback models on your real document set.
5. Benchmark different Ollama model sizes on your actual category distribution.
