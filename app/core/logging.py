import logging
import sys

import structlog


def configure_logging(level: str) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            _strip_pii,
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def _strip_pii(_, __, event_dict: dict) -> dict:
    for key in ("text", "cleaned_text", "extracted_text_preview"):
        if key in event_dict:
            event_dict[key] = "[REDACTED]"
    return event_dict

