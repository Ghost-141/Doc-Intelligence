import re
from typing import Any


def clean_ocr_pages(pages: list[dict[str, Any]]) -> tuple[str, list[str]]:
    cleaned_pages: list[str] = []
    for page in pages:
        lines = page.get("lines", [])
        rendered_lines = [_normalize_line(line.get("text", "")) for line in lines]
        rendered_lines = [line for line in rendered_lines if line]
        cleaned_pages.append(_join_lines(rendered_lines))
    full_text = "\n\n".join(page for page in cleaned_pages if page).strip()
    return full_text, cleaned_pages


def clean_text_segments(segments: list[str]) -> tuple[str, list[str]]:
    cleaned_segments = [_join_lines([_normalize_line(line) for line in segment.splitlines()]) for segment in segments]
    cleaned_segments = [segment for segment in cleaned_segments if segment]
    full_text = "\n\n".join(cleaned_segments).strip()
    return full_text, cleaned_segments


def _normalize_line(text: str) -> str:
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _join_lines(lines: list[str]) -> str:
    merged: list[str] = []
    for line in lines:
        if not line:
            continue
        if merged and _should_merge(merged[-1], line):
            merged[-1] = f"{merged[-1].rstrip('-').strip()} {line}".strip()
        else:
            merged.append(line)
    return "\n".join(merged).strip()


def _should_merge(previous: str, current: str) -> bool:
    if previous.endswith("-"):
        return True
    if previous.endswith((".", "!", "?", ":", ";")):
        return False
    if bool(re.match(r"^(\d+[\.\)]|[-*•])\s+", current)):
        return False
    return True

