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
        self._client = httpx.AsyncClient(
            base_url=self._get_provider_base_url(),
            timeout=settings.classification_timeout_seconds,
            limits=httpx.Limits(
                max_connections=max(1, settings.ollama_max_connections),
                max_keepalive_connections=max(1, settings.ollama_max_connections),
            ),
        )

    async def classify(self, extraction: ExtractionResult) -> ClassificationResult:
        started_at = perf_counter()
        text = (extraction.cleaned_text or extraction.text)[: self.settings.text_snippet_limit]
        chunks = self._build_vote_chunks(extraction, text)
        if self.settings.classifier_provider == "ollama":
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
        chunk_inputs = list(chunks or [text])
        chunk_started_at = perf_counter()
        analyses = await asyncio.gather(
            *(self._analyze_chunk(chunk, index + 1) for index, chunk in enumerate(chunk_inputs))
        )
        logger.info(
            "classification_stage_timing",
            stage=f"{self.settings.classifier_provider}_chunk_vote",
            latency_ms=round((perf_counter() - chunk_started_at) * 1000, 2),
            chunk_count=len(chunk_inputs),
        )
        return self._build_vote_result(analyses, text)

    def _build_vote_chunks(self, extraction: ExtractionResult, text: str) -> list[str]:
        page_chunks = self._build_chunks(extraction)
        if len(page_chunks) >= 3:
            return self._select_representative_chunks(page_chunks, max_chunks=3)
        if len(page_chunks) == 2:
            return page_chunks

        paragraph_chunks = self._build_first_page_paragraph_chunks(extraction)
        if len(paragraph_chunks) >= 2:
            return paragraph_chunks[:3]

        if text:
            split_text_chunks = self._split_text_for_voting(text, max_chunks=3)
            if split_text_chunks:
                return split_text_chunks

        return page_chunks or ([text] if text else [])

    def _select_representative_chunks(
        self, chunks: Sequence[str], max_chunks: int = 3
    ) -> list[str]:
        if len(chunks) <= max_chunks:
            return list(chunks)
        if max_chunks == 1:
            return [chunks[0]]
        indexes = [0, len(chunks) // 2, len(chunks) - 1] if max_chunks == 3 else [
            round(index * (len(chunks) - 1) / max(1, max_chunks - 1))
            for index in range(max_chunks)
        ]
        selected: list[str] = []
        seen_indexes: set[int] = set()
        for index in indexes:
            if index in seen_indexes:
                continue
            seen_indexes.add(index)
            selected.append(chunks[index])
        return selected

    def _split_text_for_voting(self, text: str, max_chunks: int = 3) -> list[str]:
        normalized_text = text.strip()
        if not normalized_text:
            return []
        raw_parts = [
            part.strip()
            for part in re.split(r"\f|\n\s*\n", normalized_text)
            if part and part.strip()
        ]
        if len(raw_parts) <= 1:
            raw_parts = [line.strip() for line in normalized_text.splitlines() if line.strip()]
        if len(raw_parts) <= 1:
            return _split_large_paragraph(
                normalized_text[: self.settings.text_snippet_limit],
                target_chars=max(400, min(self.settings.text_snippet_limit, len(normalized_text) // max(1, max_chunks))),
                min_chars=120,
            )[:max_chunks] or [normalized_text[: self.settings.text_snippet_limit]]

        target_chars = max(
            400,
            min(self.settings.text_snippet_limit, max(600, len(normalized_text) // max(1, max_chunks))),
        )
        min_chars = 120
        chunks: list[str] = []
        current: list[str] = []
        current_size = 0
        for part in raw_parts:
            if current and current_size + len(part) > target_chars and len(chunks) < max_chunks - 1:
                chunks.append("\n\n".join(current).strip()[: self.settings.text_snippet_limit])
                current = []
                current_size = 0
            current.append(part)
            current_size += len(part)
        if current:
            chunks.append("\n\n".join(current).strip()[: self.settings.text_snippet_limit])
        return _merge_small_chunks(chunks, min_chars=min_chars, target_chars=target_chars)[:max_chunks]

    def _build_vote_result(self, analyses: Sequence[dict], text: str) -> ClassificationResult:
        normalized = [self._normalize_analysis_result(item) for item in analyses]
        valid = [item for item in normalized if item["category"] in self.settings.category_list]
        if not valid:
            return self._heuristic_classify(text)

        aggregate: dict[str, dict[str, float]] = {}
        for item in valid:
            stats = aggregate.setdefault(
                item["category"],
                {"count": 0.0, "score": 0.0, "max_confidence": 0.0},
            )
            stats["count"] += 1
            stats["score"] += item["confidence"]
            stats["max_confidence"] = max(stats["max_confidence"], item["confidence"])

        ranked = sorted(
            aggregate.items(),
            key=lambda item: (item[1]["count"], item[1]["score"], item[1]["max_confidence"]),
            reverse=True,
        )
        best_category, best_stats = ranked[0]
        average_confidence = best_stats["score"] / max(best_stats["count"], 1.0)

        heuristic = self._heuristic_classify(text)
        if best_stats["count"] >= 2:
            if (
                best_category == "other"
                and heuristic.category != "other"
                and heuristic.confidence >= 0.75
            ):
                heuristic.rationale = "Heuristic override after majority vote returned 'other'."
                return heuristic
            return ClassificationResult(
                category=best_category,
                confidence=min(1.0, average_confidence),
                rationale="Three-chunk majority-vote classification.",
                provider=self.settings.classifier_provider,
                model=self.settings.classifier_model,
                candidates=self.settings.category_list,
            )

        if (
            heuristic.category != "other"
            and heuristic.confidence >= 0.75
            and (
                best_category == "other"
                or heuristic.confidence >= min(0.95, average_confidence + 0.15)
            )
        ):
            heuristic.rationale = "Heuristic tie-break fallback for split chunk votes."
            return heuristic

        return ClassificationResult(
            category=best_category,
            confidence=min(1.0, average_confidence),
            rationale="Three-chunk vote tie-break by summed confidence.",
            provider=self.settings.classifier_provider,
            model=self.settings.classifier_model,
            candidates=self.settings.category_list,
        )

    def _normalize_analysis_result(self, payload: dict) -> dict[str, float | str]:
        category = str(payload.get("category", "other")).strip().lower()
        if category not in self.settings.category_list:
            category = "other"
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "category": category,
            "confidence": min(max(confidence, 0.0), 1.0),
        }

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

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(httpx.HTTPError), reraise=True)
    async def _chat_json(self, system_prompt: str, user_prompt: str, schema: dict, max_tokens: int) -> dict:
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

    def _get_provider_base_url(self) -> str:
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
