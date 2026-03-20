from collections.abc import Iterable
from time import perf_counter

from fastapi import Request
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

registry = CollectorRegistry()

http_requests_total = Counter(
    "docintel_http_requests_total",
    "Total HTTP requests handled by the FastAPI app.",
    labelnames=("method", "path", "status_code"),
    registry=registry,
)

http_request_duration_seconds = Histogram(
    "docintel_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    labelnames=("method", "path"),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=registry,
)

document_classifications_total = Counter(
    "docintel_document_classifications_total",
    "Total document classification requests completed by the app.",
    labelnames=("doc_type", "classification", "status"),
    registry=registry,
)

document_classification_duration_seconds = Histogram(
    "docintel_document_classification_duration_seconds",
    "End-to-end document classification latency in seconds.",
    labelnames=("doc_type",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=registry,
)

document_stage_duration_seconds = Histogram(
    "docintel_document_stage_duration_seconds",
    "Processing time for individual document pipeline stages.",
    labelnames=("stage", "doc_type"),
    buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=registry,
)

document_upload_size_bytes = Histogram(
    "docintel_document_upload_size_bytes",
    "Uploaded document sizes in bytes.",
    labelnames=("extension",),
    buckets=(10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000),
    registry=registry,
)

service_availability = Gauge(
    "docintel_service_available",
    "Whether a core service is available after startup. 1 means available, 0 means unavailable.",
    labelnames=("service",),
    registry=registry,
)

startup_degraded = Gauge(
    "docintel_startup_degraded",
    "Whether application startup completed in degraded mode. 1 means degraded.",
    registry=registry,
)


def render_metrics() -> Response:
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


def observe_http_request(method: str, path: str, status_code: int, duration_seconds: float) -> None:
    status_code_label = str(status_code)
    http_requests_total.labels(
        method=method,
        path=path,
        status_code=status_code_label,
    ).inc()
    http_request_duration_seconds.labels(
        method=method,
        path=path,
    ).observe(duration_seconds)


def observe_document_classification(
    *,
    doc_type: str,
    classification: str,
    duration_seconds: float,
) -> None:
    document_classifications_total.labels(
        doc_type=doc_type,
        classification=classification,
        status="success",
    ).inc()
    document_classification_duration_seconds.labels(doc_type=doc_type).observe(duration_seconds)


def observe_document_stage_duration(*, stage: str, doc_type: str, duration_seconds: float) -> None:
    document_stage_duration_seconds.labels(
        stage=stage,
        doc_type=doc_type,
    ).observe(duration_seconds)


def observe_document_failure(doc_type: str) -> None:
    document_classifications_total.labels(
        doc_type=doc_type,
        classification="unknown",
        status="failed",
    ).inc()


def observe_upload_size(extension: str, size_bytes: int) -> None:
    document_upload_size_bytes.labels(extension=extension or "unknown").observe(max(size_bytes, 0))


def update_service_availability(startup_errors: dict[str, str], available_services: Iterable[str]) -> None:
    available = set(available_services)
    service_availability.labels(service="ocr").set(1 if "ocr" in available else 0)
    service_availability.labels(service="classifier").set(1 if "classifier" in available else 0)
    startup_degraded.set(1 if startup_errors else 0)


def request_timer() -> float:
    return perf_counter()


def normalize_request_path(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    return request.url.path
