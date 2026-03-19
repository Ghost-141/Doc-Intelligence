import asyncio
import json
import re
from collections.abc import Sequence
from time import perf_counter

import httpx
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.core.config import Settings
from app.schemas.document import ClassificationResult, ExtractionResult, ModelHealth

logger = structlog.get_logger(__name__)

CLASSIFICATION_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["category", "confidence"],
    "additionalProperties": False,
}


class ClassificationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._chunk_semaphore = asyncio.Semaphore(max(1, settings.classification_max_parallel_chunks))
        headers = {}
        if settings.classifier_provider == "vllm" and settings.vllm_api_key:
            headers["Authorization"] = f"Bearer {settings.vllm_api_key}"
        self._client = httpx.AsyncClient(
            base_url=self._get_provider_base_url(),
            timeout=settings.classification_timeout_seconds,
            limits=httpx.Limits(
                max_connections=max(1, settings.ollama_max_connections),
                max_keepalive_connections=max(1, settings.ollama_max_connections),
            ),
            headers=headers,
        )

    async def classify(self, extraction: ExtractionResult) -> ClassificationResult:
        started_at = perf_counter()
        text = (extraction.cleaned_text or extraction.text)[: self.settings.text_snippet_limit]
        chunks = self._build_chunks(extraction)
        if self.settings.classifier_provider in {"ollama", "vllm"}:
            result = await self._classify_with_ollama(text, chunks, extraction)
        else:
            result = self._heuristic_classify(text)
        logger.info(
            "classification_stage_timing",
            stage="classification_total",
            latency_ms=round((perf_counter() - started_at) * 1000, 2),
            chunk_count=len(chunks),
            provider=self.settings.classifier_provider,
        )
        return result

    async def warmup(self) -> None:
        started_at = perf_counter()
        try:
            await self._chat_json(
                system_prompt="Classify the document into exactly one allowed category and return only valid JSON.",
                user_prompt=f"Allowed categories: {', '.join(self.settings.category_list)}\nChunk text:\nwarmup",
                schema=CLASSIFICATION_OUTPUT_SCHEMA,
                max_tokens=self.settings.classification_chunk_max_tokens,
            )
            logger.info(
                "classifier_warmup_completed",
                provider=self.settings.classifier_provider,
                model=self.settings.classifier_model,
                latency_ms=round((perf_counter() - started_at) * 1000, 2),
            )
        except Exception as exc:
            logger.warning(
                "classifier_warmup_failed",
                provider=self.settings.classifier_provider,
                model=self.settings.classifier_model,
                error=str(exc),
            )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_health(self) -> ModelHealth:
        details = self._base_health_details()
        try:
            if self.settings.classifier_provider == "vllm":
                return await self._get_vllm_health(details)
            return await self._get_ollama_health(details)
        except Exception as exc:
            details["error"] = str(exc)
            return ModelHealth(
                name=self.settings.classifier_model,
                type="classifier",
                status="unreachable",
                uses_gpu=None,
                details=details,
            )

    def _build_chunks(self, extraction: ExtractionResult) -> list[str]:
        segments = [segment.strip() for segment in extraction.segments if segment and segment.strip()]
        if not segments:
            text = extraction.cleaned_text or extraction.text
            return [text[: self.settings.text_snippet_limit]] if text else []
        chunks: list[str] = []
        step = max(1, self.settings.classification_chunk_pages)
        for start in range(0, len(segments), step):
            page_group = segments[start : start + step]
            rendered = "\n\n".join(
                f"[Page {start + offset + 1}]\n{page[: self.settings.text_snippet_limit]}"
                for offset, page in enumerate(page_group)
            ).strip()
            if rendered:
                chunks.append(rendered)
        return chunks

    async def _classify_with_ollama(
        self, text: str, chunks: Sequence[str], extraction: ExtractionResult
    ) -> ClassificationResult:
        early_result = await self._classify_from_first_page_paragraphs(extraction)
        if early_result is not None:
            return early_result
        chunk_inputs = list(chunks or [text])
        chunk_started_at = perf_counter()
        analyses = await asyncio.gather(
            *(self._analyze_chunk(chunk, index + 1) for index, chunk in enumerate(chunk_inputs))
        )
        logger.info(
            "classification_stage_timing",
            stage=f"{self.settings.classifier_provider}_chunk_analysis",
            latency_ms=round((perf_counter() - chunk_started_at) * 1000, 2),
            chunk_count=len(chunk_inputs),
        )
        if len(analyses) == 1:
            return self._build_ollama_result(analyses[0])
        synthesis_started_at = perf_counter()
        final = await self._chat_json(
            system_prompt="Classify the document into exactly one allowed category and return only valid JSON.",
            user_prompt=self._build_synthesis_prompt(analyses),
            schema=CLASSIFICATION_OUTPUT_SCHEMA,
            max_tokens=self.settings.classification_final_max_tokens,
        )
        logger.info(
            "classification_stage_timing",
            stage=f"{self.settings.classifier_provider}_synthesis",
            latency_ms=round((perf_counter() - synthesis_started_at) * 1000, 2),
            chunk_count=len(chunk_inputs),
        )
        return self._build_ollama_result(final)

    async def _classify_from_first_page_paragraphs(
        self, extraction: ExtractionResult
    ) -> ClassificationResult | None:
        paragraph_chunks = self._build_first_page_paragraph_chunks(extraction)
        if not paragraph_chunks:
            return None
        started_at = perf_counter()
        batch_size = max(1, self.settings.classification_first_page_batch_size)
        collected: list[dict] = []
        for batch_index, start in enumerate(range(0, len(paragraph_chunks), batch_size), start=1):
            batch = paragraph_chunks[start : start + batch_size]
            analyses = await asyncio.gather(
                *(
                    self._analyze_chunk(
                        chunk,
                        index + 1,
                        chunk_kind="first_page_paragraph",
                    )
                    for index, chunk in enumerate(batch, start=start)
                )
            )
            collected.extend(analyses)
            decision = self._select_early_exit_result(collected)
            logger.info(
                "classification_stage_timing",
                stage=f"{self.settings.classifier_provider}_first_page_probe",
                latency_ms=round((perf_counter() - started_at) * 1000, 2),
                processed_chunks=len(collected),
                total_candidate_chunks=len(paragraph_chunks),
                batch_index=batch_index,
                early_exit=decision is not None,
            )
            if decision is not None:
                return decision
        return None

    async def _analyze_chunk(
        self, chunk: str, chunk_index: int, chunk_kind: str = "document_chunk"
    ) -> dict:
        prompt = (
            f"You are reviewing one {chunk_kind} from a larger document.\n"
            f"Allowed categories: {', '.join(self.settings.category_list)}\n"
            f"The OCR text may contain these languages: {', '.join(self.settings.ocr_target_language_list)}.\n"
            "Choose exactly one category for this chunk.\n"
            "Return JSON only with keys: category, confidence.\n"
            "Do not include explanations, summaries, or extra keys.\n"
            f"Chunk index: {chunk_index}\n"
            f"Chunk text:\n{chunk}"
        )
        async with self._chunk_semaphore:
            return await self._chat_json(
                system_prompt="Classify the chunk into exactly one allowed category and return only valid JSON.",
                user_prompt=prompt,
                schema=CLASSIFICATION_OUTPUT_SCHEMA,
                max_tokens=self.settings.classification_chunk_max_tokens,
            )

    def _build_synthesis_prompt(self, analyses: Sequence[dict]) -> str:
        return (
            "You are classifying a complete document using analyses from 1-2 page chunks.\n"
            f"Allowed categories: {', '.join(self.settings.category_list)}\n"
            f"The OCR text may contain these languages: {', '.join(self.settings.ocr_target_language_list)}.\n"
            "Review the chunk predictions and pick exactly one final category.\n"
            "Prefer the category that best matches the whole document.\n"
            "Return JSON only with keys: category, confidence.\n"
            f"Chunk analyses:\n{json.dumps(list(analyses), ensure_ascii=True, indent=2)}"
        )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(httpx.HTTPError), reraise=True)
    async def _chat_json(self, system_prompt: str, user_prompt: str, schema: dict, max_tokens: int) -> dict:
        if self.settings.classifier_provider == "vllm":
            return await self._chat_json_vllm(system_prompt, user_prompt, schema, max_tokens)
        response = await self._client.post(
            "/api/chat",
            json={
                "model": self.settings.classifier_model,
                "stream": False,
                "format": schema,
                "keep_alive": self.settings.ollama_keep_alive,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {"temperature": 0.0, "num_predict": max(8, max_tokens)},
            },
        )
        response.raise_for_status()
        payload = response.json()
        return _parse_ollama_json_payload(payload)

    async def _chat_json_vllm(self, system_prompt: str, user_prompt: str, schema: dict, max_tokens: int) -> dict:
        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self.settings.classifier_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                "max_tokens": max(8, max_tokens),
                "guided_json": schema,
            },
        )
        response.raise_for_status()
        payload = response.json()
        return _parse_vllm_json_payload(payload)

    def _get_provider_base_url(self) -> str:
        if self.settings.classifier_provider == "vllm":
            return self.settings.vllm_base_url.rstrip("/")
        return self.settings.ollama_base_url.rstrip("/")

    def _base_health_details(self) -> dict:
        return {
            "provider": self.settings.classifier_provider,
            "base_url": self._get_provider_base_url(),
            "max_connections": self.settings.ollama_max_connections,
            "max_parallel_chunks": self.settings.classification_max_parallel_chunks,
        }

    async def _get_ollama_health(self, details: dict) -> ModelHealth:
        response = await self._client.get("/api/ps")
        response.raise_for_status()
        payload = response.json()
        loaded_models = payload.get("models", []) if isinstance(payload, dict) else []
        current = next(
            (
                model
                for model in loaded_models
                if model.get("name") == self.settings.classifier_model
            ),
            None,
        )
        uses_gpu = _ollama_model_uses_gpu(current)
        if current:
            details["loaded_model"] = current.get("name", self.settings.classifier_model)
            details["ollama_model"] = current
            return ModelHealth(
                name=self.settings.classifier_model,
                type="classifier",
                status="loaded",
                uses_gpu=uses_gpu,
                details=details,
            )
        return ModelHealth(
            name=self.settings.classifier_model,
            type="classifier",
            status="ready",
            uses_gpu=uses_gpu,
            details=details,
        )

    async def _get_vllm_health(self, details: dict) -> ModelHealth:
        response = await self._client.get("/models")
        response.raise_for_status()
        payload = response.json()
        models = payload.get("data", []) if isinstance(payload, dict) else []
        current = next(
            (
                model
                for model in models
                if model.get("id") == self.settings.classifier_model
            ),
            None,
        )
        if current:
            details["loaded_model"] = current.get("id", self.settings.classifier_model)
            details["vllm_model"] = current
            return ModelHealth(
                name=self.settings.classifier_model,
                type="classifier",
                status="loaded",
                uses_gpu=True,
                details=details,
            )
        return ModelHealth(
            name=self.settings.classifier_model,
            type="classifier",
            status="ready",
            uses_gpu=True,
            details=details,
        )

    def _build_ollama_result(self, payload: dict) -> ClassificationResult:
        category = str(payload.get("category", "other")).strip().lower()
        if category not in self.settings.category_list:
            category = "other"
        try:
            confidence = float(payload.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(max(confidence, 0.0), 1.0)
        return ClassificationResult(
            category=category,
            confidence=confidence,
            rationale=f"Structured {self.settings.classifier_provider} classification.",
            provider=self.settings.classifier_provider,
            model=self.settings.classifier_model,
            candidates=self.settings.category_list,
        )

    def _build_first_page_paragraph_chunks(self, extraction: ExtractionResult) -> list[str]:
        raw_parts = self._extract_first_page_meaningful_parts(extraction)
        if not raw_parts:
            return []
        target_chars = max(250, self.settings.classification_first_page_target_chars)
        min_chars = max(80, self.settings.classification_first_page_min_chars)
        chunks: list[str] = []
        current: list[str] = []
        current_size = 0
        for part in raw_parts:
            if len(part) > int(target_chars * 1.2):
                if current:
                    chunks.append("\n".join(current).strip())
                    current = []
                    current_size = 0
                chunks.extend(_split_large_paragraph(part, target_chars, min_chars))
                continue
            if current and current_size + len(part) > target_chars:
                chunks.append("\n".join(current).strip())
                current = []
                current_size = 0
            current.append(part)
            current_size += len(part)
        if current:
            chunks.append("\n".join(current).strip())
        chunks = _merge_small_chunks(chunks, min_chars=min_chars, target_chars=target_chars)
        return [
            chunk[: self.settings.text_snippet_limit]
            for chunk in chunks[: self.settings.classification_first_page_max_chunks]
            if chunk
        ]

    def _extract_first_page_meaningful_parts(self, extraction: ExtractionResult) -> list[str]:
        if extraction.structured_pages:
            first_page = extraction.structured_pages[0]
            lines = [
                line.text.strip()
                for line in first_page.lines
                if line.text and line.text.strip()
            ]
            merged = _merge_ocr_lines_to_paragraphs(lines)
            if merged:
                return merged
        first_page = ""
        if extraction.segments:
            first_page = extraction.segments[0]
        elif extraction.cleaned_text or extraction.text:
            first_page = (extraction.cleaned_text or extraction.text).split("\f", 1)[0]
        first_page = first_page.strip()
        if not first_page:
            return []
        raw_parts = [
            part.strip()
            for part in re.split(r"\n\s*\n", first_page)
            if part and part.strip()
        ]
        if len(raw_parts) > 1:
            return raw_parts
        line_parts = [line.strip() for line in first_page.splitlines() if line and line.strip()]
        return _merge_ocr_lines_to_paragraphs(line_parts)

    def _select_early_exit_result(self, analyses: Sequence[dict]) -> ClassificationResult | None:
        if not analyses:
            return None
        aggregate: dict[str, dict[str, float]] = {}
        for item in analyses:
            category = str(item.get("category", "")).strip().lower()
            if category not in self.settings.category_list:
                continue
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            stats = aggregate.setdefault(
                category,
                {"count": 0.0, "score": 0.0, "max_confidence": 0.0},
            )
            stats["count"] += 1
            stats["score"] += max(confidence, 0.0)
            stats["max_confidence"] = max(stats["max_confidence"], confidence)
        if not aggregate:
            return None
        best_category, best_stats = max(
            aggregate.items(),
            key=lambda item: (item[1]["score"], item[1]["count"], item[1]["max_confidence"]),
        )
        average_confidence = best_stats["score"] / max(best_stats["count"], 1.0)
        if (
            best_stats["max_confidence"] < self.settings.classification_early_exit_confidence
            and not (
                best_stats["count"] >= 2
                and average_confidence >= self.settings.classification_early_exit_confidence
            )
        ):
            return None
        return ClassificationResult(
            category=best_category,
            confidence=min(1.0, average_confidence),
            rationale="First-page paragraph early-exit classification.",
            provider=self.settings.classifier_provider,
            model=self.settings.classifier_model,
            candidates=self.settings.category_list,
        )

    def _heuristic_classify(self, text: str) -> ClassificationResult:
        keywords: dict[str, tuple[str, ...]] = {
            "invoice": ("invoice", "total due", "bill to", "payment terms"),
            "receipt": ("receipt", "cashier", "change due"),
            "contract": ("agreement", "terms and conditions", "effective date"),
            "resume": ("experience", "education", "skills"),
            "id_document": ("passport", "date of birth", "nid", "identification"),
            "medical_record": ("patient", "diagnosis", "prescription"),
            "bank_statement": ("account number", "balance", "transaction"),
            "report": ("summary", "analysis", "findings"),
            "letter": ("dear", "regards", "sincerely"),
        }
        normalized = text.lower()
        best_category = "other"
        best_hits = 0
        for category in self.settings.category_list:
            hits = sum(1 for keyword in keywords.get(category, ()) if keyword in normalized)
            if hits > best_hits:
                best_hits = hits
                best_category = category
        confidence = min(0.99, 0.45 + (best_hits * 0.15)) if best_hits else 0.35
        return ClassificationResult(
            category=best_category,
            confidence=confidence,
            rationale="Keyword-based fallback classification.",
            provider="heuristic",
            model=self.settings.classifier_model,
            candidates=self.settings.category_list,
        )


def _ollama_model_uses_gpu(model: dict | None) -> bool | None:
    if not model:
        return None
    processor = str(model.get("processor", "")).lower()
    if processor:
        return "gpu" in processor and "0%" not in processor
    size_vram = model.get("size_vram") or model.get("vram")
    if isinstance(size_vram, (int, float)):
        return size_vram > 0
    return None


def _parse_ollama_json_payload(payload: dict) -> dict:
    message = payload.get("message") if isinstance(payload, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise ValueError("Ollama response did not contain a string JSON payload.")
    normalized = _normalize_ollama_json_text(content)
    if not normalized:
        raise ValueError("Ollama returned an empty response for a structured classification request.")
    try:
        return json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Ollama returned non-JSON content for a structured classification request: {normalized[:200]!r}"
        ) from exc


def _normalize_ollama_json_text(content: str) -> str:
    text = content.strip()
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0).strip() if match else text


def _parse_vllm_json_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("vLLM response payload was not a JSON object.")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("vLLM response did not contain any completion choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise ValueError("vLLM response did not contain a string JSON payload.")
    normalized = _normalize_ollama_json_text(content)
    if not normalized:
        raise ValueError("vLLM returned an empty response for a structured classification request.")
    try:
        return json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"vLLM returned non-JSON content for a structured classification request: {normalized[:200]!r}"
        ) from exc


def _split_large_paragraph(paragraph: str, target_chars: int, min_chars: int) -> list[str]:
    words = paragraph.split()
    if not words:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for word in words:
        projected = current_size + len(word) + (1 if current else 0)
        if current and projected > target_chars:
            chunks.append(" ".join(current).strip())
            current = [word]
            current_size = len(word)
            continue
        current.append(word)
        current_size = projected
    if current:
        chunks.append(" ".join(current).strip())
    return _merge_small_chunks(chunks, min_chars=min_chars, target_chars=target_chars)


def _merge_ocr_lines_to_paragraphs(lines: Sequence[str]) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if current and _starts_new_paragraph(current[-1], line):
            paragraphs.append(" ".join(current).strip())
            current = [line]
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current).strip())
    return [paragraph for paragraph in paragraphs if paragraph]


def _starts_new_paragraph(previous: str, current: str) -> bool:
    if previous.endswith((".", "!", "?", ":", ";")):
        return True
    if len(previous) < 40 and len(current) < 40:
        return False
    if bool(re.match(r"^(\d+[\.\)]|[-*•])\s+", current)):
        return True
    if current.isupper() and len(current.split()) <= 8:
        return True
    return False


def _merge_small_chunks(chunks: Sequence[str], min_chars: int, target_chars: int) -> list[str]:
    merged: list[str] = []
    buffer = ""
    for chunk in chunks:
        text = chunk.strip()
        if not text:
            continue
        candidate = f"{buffer}\n{text}".strip() if buffer else text
        if len(candidate) < min_chars:
            buffer = candidate
            continue
        if buffer and len(candidate) <= target_chars * 2:
            merged.append(candidate)
            buffer = ""
            continue
        if buffer:
            merged.append(buffer)
            buffer = ""
        merged.append(text)
    if buffer:
        if merged and len(buffer) < min_chars:
            merged[-1] = f"{merged[-1]}\n{buffer}".strip()
        else:
            merged.append(buffer)
    return merged
