from abc import ABC, abstractmethod
from io import BytesIO
import os
from time import perf_counter
from typing import Any

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR
import structlog

from app.core.config import Settings
from app.schemas.document import ExtractionResult, ModelHealth
from app.services.text_cleaning import clean_ocr_pages

logger = structlog.get_logger(__name__)


class OCRBackend(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def extract_image(self, file_bytes: bytes) -> ExtractionResult:
        raise NotImplementedError

    @abstractmethod
    def extract_pdf(self, file_bytes: bytes) -> ExtractionResult:
        raise NotImplementedError

    @abstractmethod
    def extract_simple_image(self, file_bytes: bytes) -> ExtractionResult:
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_health(self) -> list[ModelHealth]:
        raise NotImplementedError


class AdaptivePaddleOCRBackend(OCRBackend):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        runtime = _get_ocr_runtime_info(settings)
        self._profile = _select_ocr_profile(settings, runtime)
        ocr_kwargs = {
            "use_angle_cls": settings.ocr_detect_orientation,
            "device": self._profile["device"],
            "text_detection_model_name": self._profile["text_detection_model_name"],
            "text_recognition_model_name": self._profile["text_recognition_model_name"],
        }
        if not (
            self._profile["text_detection_model_name"]
            or self._profile["text_recognition_model_name"]
        ):
            ocr_kwargs["lang"] = settings.ocr_language
        self._ocr = PaddleOCR(
            **ocr_kwargs,
        )
        logger.info(
            "ocr_backend_selected",
            selected_path=self._profile["path"],
            requested_device=settings.ocr_device,
            active_device=self._profile["device"],
            cuda_available=runtime["cuda_available"],
            text_detection_model_name=self._profile["text_detection_model_name"],
            text_recognition_model_name=self._profile["text_recognition_model_name"],
        )

    def extract_image(self, file_bytes: bytes) -> ExtractionResult:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        return self._run_ocr_pages([image], source="paddleocr_image")

    def extract_simple_image(self, file_bytes: bytes) -> ExtractionResult:
        return self.extract_image(file_bytes)

    def extract_pdf(self, file_bytes: bytes) -> ExtractionResult:
        pdf = pdfium.PdfDocument(file_bytes)
        images = [
            pdf[index].render(scale=2.0).to_pil().convert("RGB")
            for index in range(len(pdf))
        ]
        return self._run_ocr_pages(images, source="paddleocr_pdf")

    def warmup(self) -> None:
        started_at = perf_counter()
        warmup_image = Image.new("RGB", (64, 64), color="white")
        list(self._ocr.predict(np.asarray(warmup_image)))
        logger.info(
            "ocr_warmup_completed",
            latency_ms=round((perf_counter() - started_at) * 1000, 2),
            ocr_device=self._profile["device"],
            ocr_path=self._profile["path"],
        )

    def _run_ocr_pages(
        self, images: list[Image.Image], source: str
    ) -> ExtractionResult:
        structured_pages: list[dict[str, Any]] = []
        for page_number, image in enumerate(images, start=1):
            started_at = perf_counter()
            result = list(self._ocr.predict(np.asarray(image)))
            lines = _extract_ocr_lines(result)
            logger.info(
                "ocr_stage_timing",
                stage="paddleocr_predict",
                page_number=page_number,
                latency_ms=round((perf_counter() - started_at) * 1000, 2),
                ocr_device=self._profile["device"],
                ocr_path=self._profile["path"],
            )
            lines.sort(
                key=lambda item: (
                    _line_top(item["bbox"]),
                    _line_left(item["bbox"]),
                )
            )
            structured_pages.append(
                {
                    "page_number": page_number,
                    "lines": lines,
                    "language_targets": self.settings.ocr_target_language_list,
                }
            )
        return _build_extraction_result(
            settings=self.settings,
            structured_pages=structured_pages,
            source=source,
            profile=self._profile,
        )

    def get_health(self) -> list[ModelHealth]:
        runtime = _get_ocr_runtime_info(self.settings)
        return [
            ModelHealth(
                name="PaddleOCR",
                type="ocr_primary",
                status="loaded",
                uses_gpu=self._profile["uses_gpu"],
                details={
                    **runtime,
                    "selected_path": self._profile["path"],
                    "active_device": self._profile["device"],
                    "uses_gpu": self._profile["uses_gpu"],
                    "text_detection_model_name": self._profile[
                        "text_detection_model_name"
                    ],
                    "text_recognition_model_name": self._profile[
                        "text_recognition_model_name"
                    ],
                    "gpu_detection_model_name": self.settings.ocr_gpu_detection_model,
                    "gpu_recognition_model_name": self.settings.ocr_gpu_recognition_model,
                    "cpu_detection_model_name": self.settings.ocr_cpu_detection_model,
                    "cpu_recognition_model_name": self.settings.ocr_cpu_recognition_model,
                },
            )
        ]


def build_ocr_backend(settings: Settings) -> OCRBackend:
    return AdaptivePaddleOCRBackend(settings)


def _build_extraction_result(
    settings: Settings,
    structured_pages: list[dict[str, Any]],
    source: str,
    profile: dict[str, Any],
) -> ExtractionResult:
    cleaned_text, cleaned_segments = clean_ocr_pages(structured_pages)
    return ExtractionResult(
        text=cleaned_text,
        cleaned_text=cleaned_text,
        source=source,
        segments=cleaned_segments,
        structured_pages=structured_pages,
        metadata={
            "pages": len(structured_pages),
            "lines": sum(len(page["lines"]) for page in structured_pages),
            "ocr_language": settings.ocr_language,
            "ocr_target_languages": settings.ocr_target_language_list,
            "ocr_backend": "paddleocr",
            "ocr_path": profile["path"],
            "ocr_device": profile["device"],
            "text_detection_model_name": profile["text_detection_model_name"],
            "text_recognition_model_name": profile["text_recognition_model_name"],
        },
    )


def _select_ocr_profile(
    settings: Settings, runtime: dict[str, Any]
) -> dict[str, Any]:
    requested_device = settings.ocr_device.lower()
    wants_cpu = requested_device == "cpu"
    wants_gpu = requested_device.startswith("gpu")

    if wants_cpu:
        return _build_profile(
            path="cpu",
            device="cpu",
            uses_gpu=False,
            detection_model=settings.ocr_cpu_detection_model,
            recognition_model=settings.ocr_cpu_recognition_model,
        )

    if runtime["cuda_available"]:
        return _build_profile(
            path="gpu",
            device="gpu:0",
            uses_gpu=True,
            detection_model=settings.ocr_gpu_detection_model,
            recognition_model=settings.ocr_gpu_recognition_model,
        )

    if wants_gpu:
        logger.warning(
            "ocr_gpu_unavailable_falling_back_to_cpu",
            requested_device=settings.ocr_device,
        )

    return _build_profile(
        path="cpu",
        device="cpu",
        uses_gpu=False,
        detection_model=settings.ocr_cpu_detection_model,
        recognition_model=settings.ocr_cpu_recognition_model,
    )


def _build_profile(
    *,
    path: str,
    device: str,
    uses_gpu: bool,
    detection_model: str,
    recognition_model: str,
) -> dict[str, Any]:
    return {
        "path": path,
        "device": device,
        "uses_gpu": uses_gpu,
        "text_detection_model_name": detection_model,
        "text_recognition_model_name": recognition_model,
    }


def _get_ocr_runtime_info(settings: Settings) -> dict[str, Any]:
    try:
        import paddle
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "paddlepaddle is required for OCR backends. Install paddlepaddle into the active environment."
        ) from exc
    return {
        "requested_device": settings.ocr_device,
        "cuda_available": _paddle_cuda_available(paddle),
    }


def _paddle_cuda_available(paddle_module: Any) -> bool:
    device_module = getattr(paddle_module, "device", None)
    if device_module and hasattr(device_module, "is_compiled_with_cuda"):
        try:
            return bool(device_module.is_compiled_with_cuda())
        except TypeError:
            pass
    if hasattr(paddle_module, "is_compiled_with_cuda"):
        return bool(paddle_module.is_compiled_with_cuda())
    return False


def _extract_ocr_lines(results: list[Any]) -> list[dict[str, Any]]:
    lines = _extract_paddle3_ocr_lines(results)
    if lines:
        return lines
    return _extract_legacy_tuple_lines(results)


def _extract_paddle3_ocr_lines(results: list[Any]) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for result in results:
        payload = _extract_result_payload(result)
        if not isinstance(payload, dict):
            continue
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []
        polys = payload.get("rec_polys") or payload.get("dt_polys") or []
        for index, text in enumerate(texts):
            text_value = str(text).strip()
            if not text_value:
                continue
            confidence = None
            if index < len(scores):
                try:
                    confidence = float(scores[index])
                except (TypeError, ValueError):
                    confidence = None
            bbox = []
            if index < len(polys):
                bbox = _coerce_polygon(polys[index])
            lines.append(
                {
                    "text": text_value,
                    "confidence": (
                        round(confidence, 4) if confidence is not None else None
                    ),
                    "bbox": bbox,
                }
            )
    return lines


def _extract_result_payload(result: Any) -> dict[str, Any] | None:
    if isinstance(result, dict):
        if isinstance(result.get("res"), dict):
            return result["res"]
        return result
    for attr in ("res", "json"):
        value = getattr(result, attr, None)
        if isinstance(value, dict):
            if isinstance(value.get("res"), dict):
                return value["res"]
            return value
    return None


def _extract_legacy_tuple_lines(results: list[Any]) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for block in results or []:
        if isinstance(block, (list, tuple)):
            for line in block:
                parsed = _parse_legacy_tuple_line(line)
                if parsed:
                    lines.append(parsed)
    return lines


def _parse_legacy_tuple_line(line: Any) -> dict[str, Any] | None:
    if not isinstance(line, (list, tuple)) or len(line) < 2:
        return None
    polygon = line[0]
    rec = line[1]
    if not isinstance(rec, (list, tuple)) or not rec:
        return None
    text = str(rec[0]).strip()
    if not text:
        return None
    confidence = None
    if len(rec) > 1:
        try:
            confidence = float(rec[1])
        except (TypeError, ValueError):
            confidence = None
    return {
        "text": text,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "bbox": _coerce_polygon(polygon),
    }


def _coerce_polygon(value: Any) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value:
        if all(isinstance(item, list) and len(item) >= 2 for item in value):
            return [[float(item[0]), float(item[1])] for item in value]
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            x1, y1, x2, y2 = value
            return [
                [float(x1), float(y1)],
                [float(x2), float(y1)],
                [float(x2), float(y2)],
                [float(x1), float(y2)],
            ]
    return []


def _line_top(bbox: list[list[float]]) -> float:
    return min((point[1] for point in bbox), default=0.0)


def _line_left(bbox: list[list[float]]) -> float:
    return min((point[0] for point in bbox), default=0.0)
