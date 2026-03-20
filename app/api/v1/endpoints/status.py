from fastapi import APIRouter, Request

from app.core.config import Settings
from app.schemas.document import HealthResponse, ModelHealth

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings: Settings = request.app.state.settings
    startup_errors = getattr(request.app.state, "startup_errors", {})
    extraction_service = getattr(request.app.state, "extraction_service", None)
    classification_service = getattr(request.app.state, "classification_service", None)

    models: list[ModelHealth] = []
    if extraction_service is not None:
        models.extend(extraction_service.get_health())
    else:
        models.append(_build_unavailable_ocr_health(startup_errors.get("ocr")))

    if classification_service is not None:
        models.append(await classification_service.get_health())
    else:
        models.append(
            _build_unavailable_classifier_health(
                settings=settings,
                error=startup_errors.get("classifier"),
            )
        )

    status = "ok" if all(model.status in {"ready", "loaded"} for model in models) else "degraded"
    return HealthResponse(status=status, models=models)


def _build_unavailable_ocr_health(error: str | None) -> ModelHealth:
    return ModelHealth(
        name="PaddleOCR",
        type="ocr_primary",
        status="unavailable",
        uses_gpu=None,
        details={"error": error or "OCR service failed to initialize."},
    )


def _build_unavailable_classifier_health(
    settings: Settings, error: str | None
) -> ModelHealth:
    return ModelHealth(
        name=settings.classifier_model,
        type="classifier",
        status="unavailable",
        uses_gpu=None,
        details={
            "provider": settings.classifier_provider,
            "base_url": settings.ollama_base_url.rstrip("/"),
            "error": error or "Classification service failed to initialize.",
        },
    )
