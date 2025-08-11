"""Document upload and processing endpoints."""

from fastapi import APIRouter, Depends, File, UploadFile

from backend.app.auth import AuthUser, require_auth
from backend.app.models import (
    DocumentStatusResponse,
    DocumentUploadResponse,
    ExtractedDocument,
)
from backend.app.services.document_service import document_service
from backend.app.utils.logger import get_module_logger

# Configure logging
logger = get_module_logger(__name__)

# Create router
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...), current_user: AuthUser = Depends(require_auth)
):
    """Upload and immediately process a document, returning complete analysis."""
    return await document_service.upload_and_process_document(
        file=file, user_id=current_user.name
    )


@router.get("/status/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(document_id: str):
    """Get the processing status of a document."""
    doc = document_service.get_document_status(document_id)

    return DocumentStatusResponse(
        document_id=document_id,
        status=doc["status"],
        created_at=doc.get("upload_time", doc.get("created_at")),
        updated_at=doc["updated_at"],
        error_message=doc.get("error_message"),
    )


@router.get("/extract/{document_id}", response_model=ExtractedDocument)
async def get_extracted_document(document_id: str):
    """Get the extracted and normalized content of a processed document."""
    extracted_data = document_service.get_extracted_document(document_id)

    return ExtractedDocument(
        document_id=extracted_data["document_id"],
        content=extracted_data["content"],
        extracted_at=extracted_data["extracted_at"],
        redacted_fields=extracted_data["redacted_fields"],
    )
