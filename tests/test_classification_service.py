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
