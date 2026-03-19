from time import perf_counter

from fastapi import APIRouter, Depends, File, Request, UploadFile
import structlog

from app.api.dependency import (
    get_classification_service,
    get_extraction_service,
    get_ingestion_service,
    require_api_key,
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
    ingest_started_at = perf_counter()
    document = await ingestion_service.ingest(file)
    logger.info(
        "request_stage_timing",
        stage="ingestion",
        doc_id=document.document_id,
        filename=document.original_filename,
        latency_ms=round((perf_counter() - ingest_started_at) * 1000, 2),
    )
    extraction = extraction_service.extract(document)
    classification = await classification_service.classify(extraction)
    total_latency_ms = round((perf_counter() - started_at) * 1000, 2)
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
