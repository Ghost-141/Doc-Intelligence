from fastapi import APIRouter, Depends

from app.api.dependency import get_classification_service, get_extraction_service
from app.schemas.document import HealthResponse
from app.services.classification import ClassificationService
from app.services.extraction import ExtractionService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(
    extraction_service: ExtractionService = Depends(get_extraction_service),
    classification_service: ClassificationService = Depends(get_classification_service),
) -> HealthResponse:
    models = [*extraction_service.get_health(), await classification_service.get_health()]
    status = "ok" if all(model.status in {"ready", "loaded"} for model in models) else "degraded"
    return HealthResponse(status=status, models=models)
