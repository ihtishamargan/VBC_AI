"""Document processing service for VBC AI application."""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile, status

from backend.app.config import settings
from backend.app.models import DocumentStatus, DocumentUploadResponse
from backend.app.storage.document_storage import document_storage
from backend.app.utils.file_validation import sanitize_filename, validate_pdf_content
from backend.app.utils.logger import get_module_logger

# Configure logging
logger = get_module_logger(__name__)


class DocumentService:
    """Service class for handling document operations."""

    def __init__(self):
        """Initialize the document service."""
        # Initialize services
        self.document_parser = DocumentParser(
            max_file_size_mb=settings.max_file_size_mb
        )
        self.document_ingestion = DocumentIngestionService()

        # Ensure upload directories exist
        os.makedirs(settings.upload_dir, exist_ok=True)
        os.makedirs(settings.processed_dir, exist_ok=True)

        logger.info("DocumentService initialized successfully")

    async def upload_and_process_document(
        self, file: UploadFile, user_id: str
    ) -> DocumentUploadResponse:
        """Upload and process a document with comprehensive error handling."""
        document_id = None
        temp_file_path = None
        file_path = None

        try:
            # Rate limiting check (user-based instead of IP-based)
            recent_uploads = document_storage.get_recent_upload_count(
                user_id, minutes=60
            )
            if recent_uploads >= 10:  # Max 10 uploads per hour per user
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Upload rate limit exceeded. Please try again later.",
                )

            # Track this upload attempt
            document_storage.track_upload_attempt(user_id)

            # Validate file
            if not file.filename:
                raise HTTPException(status_code=400, detail="No filename provided")

            # Check file type
            content_type = file.content_type or mimetypes.guess_type(file.filename)[0]
            if content_type != "application/pdf":
                raise HTTPException(
                    status_code=400, detail="Only PDF files are supported"
                )

            # Read and validate file content
            content = await file.read()

            # Check file size
            if len(content) > settings.max_file_size_mb * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {settings.max_file_size_mb}MB",
                )

            # Validate PDF content
            if not validate_pdf_content(content):
                raise HTTPException(
                    status_code=400, detail="Invalid or potentially unsafe PDF file"
                )

            # Generate document ID and sanitize filename
            document_id = str(uuid.uuid4())
            safe_filename = sanitize_filename(file.filename)

            # Check for duplicate content
            content_hash = await deduplication_service.calculate_content_hash(content)
            existing_doc = await deduplication_service.find_duplicate(content_hash)

            if existing_doc:
                logger.info(
                    f"Duplicate document detected: {existing_doc['document_id']}"
                )
                return DocumentUploadResponse(
                    document_id=existing_doc["document_id"],
                    filename=existing_doc["filename"],
                    status=existing_doc["status"],
                    message="Document already exists in the system",
                    duplicate_of=existing_doc["document_id"],
                    vbc_data=existing_doc.get("vbc_data"),
                )

            # Create document record
            document_storage.create_document_record(
                document_id=document_id,
                filename=safe_filename,
                file_size=len(content),
                user_id=user_id,
            )

            # Save file to temporary location
            temp_file_path = Path(settings.upload_dir) / f"temp_{document_id}.pdf"
            with open(temp_file_path, "wb") as f:
                f.write(content)

            # Move to final location
            file_path = Path(settings.upload_dir) / f"{document_id}.pdf"
            temp_file_path.rename(file_path)
            temp_file_path = None  # Clear temp path since file was moved

            logger.info(f"File saved: {file_path}")

            # Register document with deduplication service
            await deduplication_service.register_document(
                document_id=document_id,
                content_hash=content_hash,
                filename=safe_filename,
                file_size=len(content),
            )

            # Start background processing
            asyncio.create_task(
                self._process_document_background(document_id, str(file_path))
            )

            return DocumentUploadResponse(
                document_id=document_id,
                filename=safe_filename,
                status=DocumentStatus.PROCESSING,
                message="Document uploaded successfully and processing started",
            )

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            logger.error(f"Document upload failed: {str(e)}", exc_info=True)

            # Clean up on error
            await self._cleanup_failed_upload(document_id, temp_file_path, file_path)

            # Return generic error message to avoid information disclosure
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document processing failed. Please try again or contact support.",
            )

    async def _process_document_background(self, document_id: str, file_path: str):
        """Background task to process uploaded document."""
        try:
            logger.info(f"Starting background processing for document {document_id}")

            # Update status to processing
            document_storage.update_document_status(
                document_id, DocumentStatus.PROCESSING
            )

            # Step 1: Parse the document
            pages = await self.document_parser.parse_pdf(file_path)
            logger.info(f"Parsed {len(pages)} pages from document {document_id}")

            # Step 2: Ingest document with deduplication (analyze, chunk, store vectors)
            result = await self.document_ingestion.ingest_document(
                pages=pages, document_id=document_id, check_duplicates=True
            )

            logger.info(f"Document ingestion completed for {document_id}")

            # Step 3: Store processing results
            processed_data = {
                "insights": result.get("insights", {}),
                "metadata": {
                    "pages_processed": len(pages),
                    "chunks_created": result.get("chunks_created", 0),
                    "processing_time": result.get("processing_time", 0.0),
                    "processed_at": datetime.now(),
                    "file_size": document_storage.get_document(document_id).get(
                        "file_size", 0
                    ),
                },
                "chunks": result.get("chunks", []),
                "raw_pages": pages[:5]
                if pages
                else [],  # Store first 5 pages as preview
            }

            document_storage.store_processed_data(document_id, processed_data)

            # Step 4: Update status to completed
            document_storage.update_document_status(document_id, DocumentStatus.DONE)

            logger.info(f"Document {document_id} processing completed successfully")

        except Exception as e:
            logger.error(
                f"Background processing failed for document {document_id}: {str(e)}",
                exc_info=True,
            )

            # Update status to failed
            document_storage.update_document_status(
                document_id, DocumentStatus.FAILED, error_message=str(e)
            )

    async def _cleanup_failed_upload(
        self,
        document_id: str | None,
        temp_file_path: Path | None,
        file_path: Path | None,
    ):
        """Clean up resources after failed upload."""
        # Clean up temporary file if it exists
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass

        # Clean up final file if it was saved
        if file_path and file_path.exists():
            try:
                file_path.unlink()
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up file {file_path}: {cleanup_error}")

        # Clean up document record
        if document_id:
            document_storage.delete_document(document_id)

    def get_document_status(self, document_id: str) -> dict[str, Any]:
        """Get document processing status."""
        if not document_storage.document_exists(document_id):
            raise HTTPException(status_code=404, detail="Document not found")

        return document_storage.get_document(document_id)

    def get_extracted_document(self, document_id: str) -> dict[str, Any]:
        """Get extracted document content and metadata."""
        if not document_storage.document_exists(document_id):
            raise HTTPException(status_code=404, detail="Document not found")

        doc = document_storage.get_document(document_id)
        if doc["status"] != DocumentStatus.DONE:
            raise HTTPException(status_code=400, detail="Document not yet processed")

        processed_data = document_storage.get_processed_data(document_id)
        if not processed_data:
            raise HTTPException(
                status_code=404, detail="Processed document data not found"
            )

        insights = processed_data.get("insights", {})
        metadata = processed_data.get("metadata", {})

        # Build extracted content with processing metadata
        extracted_content = {
            "summary": insights.get("summary", "No summary available"),
            "topics": insights.get("topics", []),
            "entities": insights.get("entities", []),
            "document_type": insights.get("document_type", "unknown"),
            "confidence_score": insights.get("confidence", 0.0),
            "metadata": {
                "pages_processed": metadata.get("pages_processed", 0),
                "chunks_created": metadata.get("chunks_created", 0),
                "processing_time": metadata.get("processing_time", 0.0),
                "file_size": metadata.get("file_size", doc.get("file_size", 0)),
            },
            "chunks_preview": processed_data.get("chunks", [])[
                :3
            ],  # First 3 chunks as preview
            "raw_pages_preview": processed_data.get("raw_pages", []),
        }

        # Check for potential PHI based on entities and content
        redacted_fields = []
        for entity in insights.get("entities", []):
            if entity.get("type") in ["PERSON", "EMAIL", "PHONE", "SSN"]:
                redacted_fields.append(entity["type"].lower())

        # Additional PHI check based on filename
        if "personal" in doc.get("filename", "").lower():
            redacted_fields.extend(["personal_info", "contact_details"])

        return {
            "document_id": document_id,
            "content": extracted_content,
            "extracted_at": metadata.get("processed_at", datetime.now()),
            "redacted_fields": list(set(redacted_fields)),  # Remove duplicates
        }

    def get_deduplication_stats(self) -> dict[str, Any]:
        """Get document deduplication statistics."""
        try:
            stats = deduplication_service.get_stats()
            return {
                "success": True,
                "stats": stats,
                "message": f"Currently tracking {stats['total_documents']} unique documents",
            }
        except Exception as e:
            logger.error(f"Failed to get deduplication stats: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve deduplication statistics"
            )


# Global document service instance
document_service = DocumentService()
