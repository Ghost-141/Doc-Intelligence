from contextlib import asynccontextmanager
import logging
import os
import warnings

from requests import RequestsDependencyWarning

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.security import limiter
from app.services.classification import ClassificationService
from app.services.extraction import ExtractionService
from app.services.ingestion import IngestionService

os.environ.setdefault("GLOG_minloglevel", "2")

warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
warnings.filterwarnings("ignore", message="No ccache found.*")


class _SuppressNoisyPaddleMessages(logging.Filter):
    BLOCKED_SUBSTRINGS = (
        "Connectivity check to the model hoster has been skipped",
        "Checking connectivity to the model hosters, this may take a while",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(text in message for text in self.BLOCKED_SUBSTRINGS)


logging.getLogger().addFilter(_SuppressNoisyPaddleMessages())


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.settings = settings
    app.state.ingestion_service = IngestionService(settings)
    app.state.extraction_service = ExtractionService(settings)
    app.state.extraction_service.warmup()
    app.state.classification_service = ClassificationService(settings)
    try:
        yield
    finally:
        await app.state.classification_service.aclose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.include_router(api_router, prefix="/api")
    return app


app = create_app()
