from fastapi import APIRouter

from app.api.v1.endpoints import document, status

api_router = APIRouter()
api_router.include_router(status.router, tags=["status"])
api_router.include_router(document.router, prefix="/v1/documents", tags=["documents"])

