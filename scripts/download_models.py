import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OCR and classification model assets.")
    parser.add_argument("--ocr-model-id", default="PaddlePaddle/PaddleOCR-VL-1.5", help="Hugging Face OCR model repo id.")
    parser.add_argument("--ocr-dest", default="models/PaddleOCR-VL-1.5", help="Local directory for OCR weights.")
    parser.add_argument("--ollama-model", default="qwen2.5:3b", help="Ollama model to pull.")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR model download.")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama model pull.")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token for gated or rate-limited downloads.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.skip_ocr:
        download_hf_model(args.ocr_model_id, Path(args.ocr_dest), args.hf_token)
    if not args.skip_ollama:
        pull_ollama_model(args.ollama_model)
    return 0


def download_hf_model(model_id: str, destination: Path, token: str | None) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,
    )
    print(f"Downloaded OCR model '{model_id}' to '{destination}'.")


def pull_ollama_model(model_name: str) -> None:
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        raise RuntimeError("Ollama is not installed or not available on PATH.")
    subprocess.run([ollama_path, "pull", model_name], check=True)
    print(f"Pulled Ollama model '{model_name}'.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"download_models.py failed: {exc}", file=sys.stderr)
        raise
