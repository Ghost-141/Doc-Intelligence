#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/doc-intel-engine}"
ENV_FILE="${ENV_FILE:-$APP_DIR/.env.production}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$APP_DIR/.venv/bin/activate}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
DOC_INTEL_SERVICE="${DOC_INTEL_SERVICE:-doc-intel}"
OLLAMA_SERVICE="${OLLAMA_SERVICE:-ollama}"
NGINX_SERVICE="${NGINX_SERVICE:-nginx}"
PADDLE_PACKAGE="${PADDLE_PACKAGE:-paddlepaddle}"

cd "$APP_DIR"

if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "Virtual environment not found at $VENV_ACTIVATE" >&2
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE" >&2
  exit 1
fi

if [ -z "$UV_BIN" ]; then
  echo "uv is not installed or not available in PATH" >&2
  exit 1
fi

source "$VENV_ACTIVATE"

git pull --ff-only
"$UV_BIN" sync
"$UV_BIN" pip install --python "$APP_DIR/.venv/bin/python" "$PADDLE_PACKAGE"

sudo systemctl restart "$OLLAMA_SERVICE"
sudo systemctl restart "$DOC_INTEL_SERVICE"
sudo systemctl reload "$NGINX_SERVICE"
sudo systemctl status "$DOC_INTEL_SERVICE" --no-pager
