#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/imtiaz/Documents/Doc-Intelligence/"
VENV_ACTIVATE="$APP_DIR/.venv/bin/activate"

cd "$APP_DIR"

if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "Virtual environment not found at $VENV_ACTIVATE" >&2
  exit 1
fi

if [ ! -f "$APP_DIR/.env.production" ]; then
  echo "Missing $APP_DIR/.env.production" >&2
  exit 1
fi

source "$VENV_ACTIVATE"

git pull --ff-only
uv sync
uv pip install paddlepaddle

sudo systemctl restart ollama
sudo systemctl restart doc-intel
sudo systemctl reload nginx
sudo systemctl status doc-intel --no-pager
