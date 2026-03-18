from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from app.core.config import Settings
from app.schemas.document import DocumentIngested


class IngestionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def ingest(self, upload: UploadFile) -> DocumentIngested:
        content = await upload.read()
        if not upload.filename:
            raise HTTPException(status_code=400, detail="Filename is required.")
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        max_size = self.settings.max_upload_size_mb * 1024 * 1024
        if len(content) > max_size:
            raise HTTPException(status_code=413, detail="Uploaded file exceeds size limit.")

        document_id = uuid4().hex
        safe_name = Path(upload.filename).name
        storage_name = f"{document_id}_{safe_name}"
        file_path = self.settings.upload_dir / storage_name
        file_path.write_bytes(content)

        return DocumentIngested(
            document_id=document_id,
            original_filename=safe_name,
            storage_name=storage_name,
            media_type=upload.content_type or "application/octet-stream",
            file_path=str(file_path),
            size_bytes=len(content),
            extension=Path(safe_name).suffix.lower(),
        )

