from fastapi import Depends, Header, HTTPException, Request, status

from app.core.config import Settings, get_settings
from app.services.classification import ClassificationService
from app.services.extraction import ExtractionService
from app.services.ingestion import IngestionService


def get_settings_dependency() -> Settings:
    return get_settings()


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_extraction_service(request: Request) -> ExtractionService:
    return _require_service(
        request=request,
        state_name="extraction_service",
        service_label="Extraction",
    )


def get_classification_service(request: Request) -> ClassificationService:
    return _require_service(
        request=request,
        state_name="classification_service",
        service_label="Classification",
    )


def _require_service(request: Request, state_name: str, service_label: str):
    service = getattr(request.app.state, state_name, None)
    if service is not None:
        return service
    startup_errors = getattr(request.app.state, "startup_errors", {})
    error = startup_errors.get(
        "ocr" if state_name == "extraction_service" else "classifier"
    )
    detail = f"{service_label} service is unavailable."
    if error:
        detail = f"{detail} Startup error: {error}"
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=detail,
    )


def require_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings_dependency),
) -> None:
    if not settings.enable_api_key_auth:
        return
    if not settings.api_key_list:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key auth is enabled but no API keys are configured.",
        )
    if x_api_key not in settings.api_key_list:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key."
        )
