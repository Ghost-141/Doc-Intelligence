from time import perf_counter

from fastapi import APIRouter, Depends, File, Request, UploadFile
import structlog

from app.api.dependency import (
    get_classification_service,
    get_extraction_service,
    get_ingestion_service,
    require_api_key,
)
from app.core.metrics import (
    observe_document_classification,
    observe_document_failure,
    observe_document_stage_duration,
    observe_upload_size,
)
from app.schemas.document import DocumentResponse
from app.services.classification import ClassificationService
from app.services.extraction import ExtractionService
from app.services.ingestion import IngestionService

logger = structlog.get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/classify", response_model=DocumentResponse)
async def classify_document(
    request: Request,
    file: UploadFile = File(...),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    extraction_service: ExtractionService = Depends(get_extraction_service),
    classification_service: ClassificationService = Depends(get_classification_service),
) -> DocumentResponse:
    started_at = perf_counter()
    upload_content_type = file.content_type or "unknown"
    ingest_started_at = perf_counter()
    document = await ingestion_service.ingest(file)
    ingest_duration_seconds = perf_counter() - ingest_started_at
    observe_upload_size(document.extension or "unknown", document.size_bytes)
    observe_document_stage_duration(
        stage="ingestion",
        doc_type=document.media_type,
        duration_seconds=ingest_duration_seconds,
    )
    logger.info(
        "request_stage_timing",
        stage="ingestion",
        doc_id=document.document_id,
        filename=document.original_filename,
        latency_ms=round(ingest_duration_seconds * 1000, 2),
    )
    try:
        extraction_started_at = perf_counter()
        extraction = extraction_service.extract(document)
        extraction_duration_seconds = perf_counter() - extraction_started_at
        observe_document_stage_duration(
            stage="extraction",
            doc_type=document.media_type,
            duration_seconds=extraction_duration_seconds,
        )
        classification_started_at = perf_counter()
        classification = await classification_service.classify(extraction)
        classification_duration_seconds = perf_counter() - classification_started_at
        observe_document_stage_duration(
            stage="classification",
            doc_type=document.media_type,
            duration_seconds=classification_duration_seconds,
        )
    except Exception:
        observe_document_failure(document.media_type if "document" in locals() else upload_content_type)
        raise
    total_latency_ms = round((perf_counter() - started_at) * 1000, 2)
    observe_document_classification(
        doc_type=document.media_type,
        classification=classification.category,
        duration_seconds=total_latency_ms / 1000,
    )
    logger.info(
        "request_stage_timing",
        stage="request_total",
        doc_id=document.document_id,
        filename=document.original_filename,
        latency_ms=total_latency_ms,
        doc_type=document.media_type,
    )
    return DocumentResponse(
        doc_id=document.document_id,
        filename=document.original_filename,
        doc_type=document.media_type,
        classification=classification.category,
        confidence=classification.confidence,
        latency_ms=total_latency_ms,
        ocr_text_preview=(extraction.cleaned_text or extraction.text)[:500],
    )
