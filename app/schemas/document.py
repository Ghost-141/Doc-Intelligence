from typing import Any

from pydantic import BaseModel, Field


class OCRLine(BaseModel):
    text: str
    confidence: float | None = None
    bbox: list[list[float]] = Field(default_factory=list)


class OCRPage(BaseModel):
    page_number: int
    lines: list[OCRLine] = Field(default_factory=list)
    language_targets: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    text: str
    cleaned_text: str = ""
    source: str
    segments: list[str] = Field(default_factory=list)
    structured_pages: list[OCRPage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClassificationResult(BaseModel):
    category: str
    confidence: float
    rationale: str
    provider: str
    model: str
    candidates: list[str] = Field(default_factory=list)


class DocumentIngested(BaseModel):
    document_id: str
    original_filename: str
    storage_name: str
    media_type: str
    file_path: str
    size_bytes: int
    extension: str


class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    doc_type: str
    classification: str
    confidence: float
    latency_ms: float
    ocr_text_preview: str


class ModelHealth(BaseModel):
    name: str
    type: str
    status: str
    uses_gpu: bool | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    models: list[ModelHealth] = Field(default_factory=list)
