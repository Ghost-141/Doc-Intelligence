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
    return request.app.state.extraction_service


def get_classification_service(request: Request) -> ClassificationService:
    return request.app.state.classification_service


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
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")
