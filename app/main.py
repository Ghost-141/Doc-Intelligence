import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from time import perf_counter
import warnings

from requests import RequestsDependencyWarning

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.metrics import normalize_request_path, observe_http_request, render_metrics, request_timer, update_service_availability
from app.services.classification import ClassificationService
from app.services.extraction import ExtractionService
from app.services.ingestion import IngestionService

os.environ.setdefault("GLOG_minloglevel", "2")

warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
warnings.filterwarnings("ignore", message="No ccache found.*")

startup_logger = logging.getLogger(__name__)


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

    startup_logger.info("startup: initializing services")
    started_at = perf_counter()

    app.state.ingestion_service = IngestionService(settings)
    app.state.extraction_service = None
    app.state.classification_service = None
    app.state.startup_errors = {}

    loop = asyncio.get_event_loop()

    def _init_and_warmup_ocr() -> ExtractionService:
        svc = ExtractionService(settings)  # PaddleOCR model load happens here
        svc.warmup()
        return svc

    async def _init_and_warmup_classifier() -> ClassificationService:
        svc = ClassificationService(settings)
        await svc.warmup()
        return svc

    results = await asyncio.gather(
        loop.run_in_executor(None, _init_and_warmup_ocr),
        _init_and_warmup_classifier(),
        return_exceptions=True,
    )

    for name, result in zip(("ocr", "classifier"), results):
        if isinstance(result, Exception):
            app.state.startup_errors[name] = str(result)
            startup_logger.warning("startup: %s init/warmup failed - %s", name, result)
        else:
            startup_logger.info("startup: %s ready", name)

    app.state.extraction_service = results[0] if not isinstance(results[0], Exception) else None
    app.state.classification_service = results[1] if not isinstance(results[1], Exception) else None
    update_service_availability(
        startup_errors=app.state.startup_errors,
        available_services=(
            name
            for name, service in (
                ("ocr", app.state.extraction_service),
                ("classifier", app.state.classification_service),
            )
            if service is not None
        ),
    )

    startup_duration_ms = (perf_counter() - started_at) * 1000
    if app.state.startup_errors:
        startup_logger.warning(
            "startup: completed with degraded services in %.0f ms",
            startup_duration_ms,
        )
    else:
        startup_logger.info(
            "startup: all services ready in %.0f ms",
            startup_duration_ms,
        )

    try:
        yield
    finally:
        if app.state.classification_service is not None:
            await app.state.classification_service.aclose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    web_dir = Path(__file__).resolve().parent.parent / "frotnend"

    @app.middleware("http")
    async def prometheus_http_metrics(request: Request, call_next):
        started_at = request_timer()
        try:
            response = await call_next(request)
        except Exception:
            observe_http_request(
                method=request.method,
                path=normalize_request_path(request),
                status_code=500,
                duration_seconds=perf_counter() - started_at,
            )
            raise
        observe_http_request(
            method=request.method,
            path=normalize_request_path(request),
            status_code=response.status_code,
            duration_seconds=perf_counter() - started_at,
        )
        return response

    @app.get("/", include_in_schema=False)
    async def serve_frontend() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return render_metrics()

    app.mount("/frontend-assets", StaticFiles(directory=web_dir), name="frontend-assets")
    app.include_router(api_router, prefix="/api")
    return app


app = create_app()
