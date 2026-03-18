from io import BytesIO
from pathlib import Path
from time import perf_counter

from bs4 import BeautifulSoup
from docx import Document
from fastapi import HTTPException
from markdown import markdown
from PIL import Image
from pypdf import PdfReader
import structlog

from app.core.config import Settings
from app.schemas.document import DocumentIngested, ExtractionResult, ModelHealth
from app.services.ocr_backends import build_ocr_backend
from app.services.text_cleaning import clean_text_segments

logger = structlog.get_logger(__name__)


class ExtractionService:
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    MARKDOWN_EXTENSIONS = {".md", ".markdown"}
    TEXT_EXTENSIONS = {".txt"}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ocr_backend = build_ocr_backend(settings)

    def get_health(self) -> list[ModelHealth]:
        return self._ocr_backend.get_health()

    def warmup(self) -> None:
        self._ocr_backend.warmup()

    def extract(self, document: DocumentIngested) -> ExtractionResult:
        started_at = perf_counter()
        file_bytes = Path(document.file_path).read_bytes()
        if document.extension == ".pdf":
            extraction = self._extract_pdf_text(file_bytes)
            if len(extraction.cleaned_text or extraction.text) >= self.settings.min_direct_text_length:
                self._log_stage_timing("pdf_direct_text", started_at, document)
                return extraction
            result = self._extract_pdf_ocr(file_bytes, base_metadata=extraction.metadata)
            self._log_stage_timing("pdf_ocr_fallback", started_at, document)
            return result
        if document.extension == ".docx":
            result = self._extract_docx(file_bytes)
            self._log_stage_timing("docx_text", started_at, document)
            return result
        if document.extension in self.MARKDOWN_EXTENSIONS:
            result = self._extract_markdown(file_bytes)
            self._log_stage_timing("markdown_text", started_at, document)
            return result
        if document.extension in self.TEXT_EXTENSIONS:
            result = self._extract_plaintext(file_bytes)
            self._log_stage_timing("plain_text", started_at, document)
            return result
        if document.extension in self.IMAGE_EXTENSIONS:
            result = self._extract_image_ocr(file_bytes, document)
            self._log_stage_timing("simple_image_ocr", started_at, document)
            return result
        raise HTTPException(status_code=415, detail=f"Unsupported extension: {document.extension or 'unknown'}")

    def _extract_pdf_text(self, file_bytes: bytes) -> ExtractionResult:
        reader = PdfReader(BytesIO(file_bytes))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        full_text, cleaned_segments = clean_text_segments(pages)
        return ExtractionResult(
            text=full_text,
            cleaned_text=full_text,
            source="pdf_text",
            segments=cleaned_segments,
            metadata={"pages": len(reader.pages)},
        )

    def _extract_docx(self, file_bytes: bytes) -> ExtractionResult:
        document = Document(BytesIO(file_bytes))
        paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        full_text, cleaned_segments = clean_text_segments(_chunk_paragraphs(paragraphs))
        return ExtractionResult(
            text=full_text,
            cleaned_text=full_text,
            source="docx_text",
            segments=cleaned_segments,
            metadata={"paragraphs": len(paragraphs)},
        )

    def _extract_markdown(self, file_bytes: bytes) -> ExtractionResult:
        html = markdown(file_bytes.decode("utf-8", errors="ignore"))
        text = BeautifulSoup(html, "html.parser").get_text(separator="\n").strip()
        full_text, cleaned_segments = clean_text_segments(_chunk_text(text))
        return ExtractionResult(text=full_text, cleaned_text=full_text, source="markdown_text", segments=cleaned_segments, metadata={})

    def _extract_plaintext(self, file_bytes: bytes) -> ExtractionResult:
        text = file_bytes.decode("utf-8", errors="ignore").strip()
        full_text, cleaned_segments = clean_text_segments(_chunk_text(text))
        return ExtractionResult(text=full_text, cleaned_text=full_text, source="plain_text", segments=cleaned_segments, metadata={})

    def _extract_image_ocr(self, file_bytes: bytes, document: DocumentIngested) -> ExtractionResult:
        preprocess_started_at = perf_counter()
        prepared = _compress_image_for_ocr(
            file_bytes=file_bytes,
            max_dimension=self.settings.image_ocr_max_dimension,
            jpeg_quality=self.settings.image_ocr_jpeg_quality,
        )
        logger.info(
            "ocr_stage_timing",
            stage="image_preprocess",
            doc_id=document.document_id,
            filename=document.original_filename,
            latency_ms=round((perf_counter() - preprocess_started_at) * 1000, 2),
            original_size_bytes=len(file_bytes),
            prepared_size_bytes=len(prepared),
        )
        return self._ocr_backend.extract_simple_image(prepared)

    def _extract_pdf_ocr(self, file_bytes: bytes, base_metadata: dict) -> ExtractionResult:
        result = self._ocr_backend.extract_pdf(file_bytes)
        result.metadata["fallback_from"] = "pdf_text"
        result.metadata.update(base_metadata)
        return result

    def _log_stage_timing(self, stage: str, started_at: float, document: DocumentIngested) -> None:
        logger.info(
            "extraction_stage_timing",
            stage=stage,
            doc_id=document.document_id,
            filename=document.original_filename,
            latency_ms=round((perf_counter() - started_at) * 1000, 2),
        )


def _chunk_paragraphs(paragraphs: list[str], target_size: int = 2200) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for paragraph in paragraphs:
        current.append(paragraph)
        current_size += len(paragraph)
        if current_size >= target_size:
            chunks.append("\n".join(current))
            current = []
            current_size = 0
    if current:
        chunks.append("\n".join(current))
    return chunks


def _chunk_text(text: str, target_size: int = 2200) -> list[str]:
    if not text:
        return []
    lines = [line for line in text.splitlines() if line.strip()]
    return _chunk_paragraphs(lines, target_size=target_size) if lines else [text]


def _compress_image_for_ocr(file_bytes: bytes, max_dimension: int, jpeg_quality: int) -> bytes:
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    image.thumbnail((max_dimension, max_dimension))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return buffer.getvalue()
