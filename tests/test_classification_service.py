import asyncio

from app.core.config import Settings
from app.schemas.document import ExtractionResult
from app.services.classification import ClassificationService


def test_build_chunks_groups_two_pages() -> None:
    service = ClassificationService(Settings(CLASSIFICATION_CHUNK_PAGES=2, CLASSIFIER_PROVIDER="heuristic"))
    try:
        extraction = ExtractionResult(
            text="one\ntwo\nthree",
            cleaned_text="one\ntwo\nthree",
            source="pdf_text",
            segments=["one", "two", "three"],
            metadata={"pages": 3},
        )

        chunks = service._build_chunks(extraction)

        assert len(chunks) == 2
        assert "[Page 1]" in chunks[0]
        assert "[Page 3]" in chunks[1]
    finally:
        asyncio.run(service.aclose())


def test_build_vote_result_returns_majority_class() -> None:
    service = ClassificationService(Settings(CLASSIFIER_PROVIDER="heuristic"))
    try:
        result = service._build_vote_result(
            analyses=[
                {"category": "resume", "confidence": 0.91},
                {"category": "resume", "confidence": 0.88},
                {"category": "other", "confidence": 0.52},
            ],
            text="education skills experience projects internship",
        )

        assert result.category == "resume"
        assert "majority-vote" in result.rationale.lower()
    finally:
        asyncio.run(service.aclose())


def test_build_vote_result_uses_heuristic_on_other_majority() -> None:
    service = ClassificationService(Settings(CLASSIFIER_PROVIDER="heuristic"))
    try:
        result = service._build_vote_result(
            analyses=[
                {"category": "other", "confidence": 0.56},
                {"category": "other", "confidence": 0.54},
                {"category": "resume", "confidence": 0.81},
            ],
            text="education skills experience projects internship",
        )

        assert result.category == "resume"
        assert "heuristic" in result.rationale.lower()
    finally:
        asyncio.run(service.aclose())


def test_build_vote_chunks_splits_single_long_segment() -> None:
    service = ClassificationService(Settings(CLASSIFIER_PROVIDER="heuristic"))
    try:
        extraction = ExtractionResult(
            text="\n\n".join(
                [
                    "education skills experience projects " * 20,
                    "internship backend fastapi python " * 20,
                    "leadership research achievements " * 20,
                ]
            ),
            cleaned_text="\n\n".join(
                [
                    "education skills experience projects " * 20,
                    "internship backend fastapi python " * 20,
                    "leadership research achievements " * 20,
                ]
            ),
            source="pdf_text",
            segments=["education skills experience projects " * 60],
            metadata={"pages": 1},
        )

        chunks = service._build_vote_chunks(extraction, extraction.cleaned_text)

        assert 1 < len(chunks) <= 3
    finally:
        asyncio.run(service.aclose())
