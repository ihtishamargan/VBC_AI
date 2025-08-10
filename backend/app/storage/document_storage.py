"""Document storage management for in-memory document tracking."""

import logging
from datetime import datetime
from typing import Any

from backend.app.models import DocumentStatus

logger = logging.getLogger(__name__)


class DocumentStorage:
    """Manages in-memory document storage and tracking."""

    def __init__(self):
        """Initialize document storage."""
        # In-memory storage (replace with actual database in production)
        self.documents_db: dict[str, Any] = {}
        self.processed_documents: dict[str, Any] = {}

        # Simple rate limiting (in production, use Redis or proper rate limiter)
        self.upload_attempts: dict[str, list[datetime]] = {}  # IP -> list of timestamps

        logger.info("DocumentStorage initialized")

    def create_document_record(
        self, document_id: str, filename: str, file_size: int, user_id: str
    ) -> dict[str, Any]:
        """Create a new document record."""
        document_record = {
            "document_id": document_id,
            "filename": filename,
            "file_size": file_size,
            "status": DocumentStatus.PROCESSING,
            "upload_time": datetime.now(),
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "user_id": user_id,
            "error_message": None,
        }

        self.documents_db[document_id] = document_record
        logger.info(f"Created document record for {document_id}")
        return document_record

    def update_document_status(
        self, document_id: str, status: DocumentStatus, error_message: str | None = None
    ) -> None:
        """Update document status."""
        if document_id in self.documents_db:
            self.documents_db[document_id]["status"] = status
            self.documents_db[document_id]["updated_at"] = datetime.now()
            if error_message:
                self.documents_db[document_id]["error_message"] = error_message
            logger.info(f"Updated document {document_id} status to {status}")

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Get document record by ID."""
        return self.documents_db.get(document_id)

    def document_exists(self, document_id: str) -> bool:
        """Check if document exists."""
        return document_id in self.documents_db

    def delete_document(self, document_id: str) -> None:
        """Delete document record."""
        if document_id in self.documents_db:
            del self.documents_db[document_id]
            logger.info(f"Deleted document record {document_id}")

        if document_id in self.processed_documents:
            del self.processed_documents[document_id]
            logger.info(f"Deleted processed document data {document_id}")

    def store_processed_data(
        self, document_id: str, processed_data: dict[str, Any]
    ) -> None:
        """Store processed document data."""
        self.processed_documents[document_id] = processed_data
        logger.info(f"Stored processed data for document {document_id}")

    def get_processed_data(self, document_id: str) -> dict[str, Any] | None:
        """Get processed document data."""
        return self.processed_documents.get(document_id)

    def track_upload_attempt(self, ip_address: str) -> None:
        """Track upload attempt for rate limiting."""
        now = datetime.now()
        if ip_address not in self.upload_attempts:
            self.upload_attempts[ip_address] = []

        self.upload_attempts[ip_address].append(now)

        # Clean up old attempts (older than 1 hour)
        cutoff = datetime.now().timestamp() - 3600  # 1 hour ago
        self.upload_attempts[ip_address] = [
            attempt
            for attempt in self.upload_attempts[ip_address]
            if attempt.timestamp() > cutoff
        ]

    def get_recent_upload_count(self, ip_address: str, minutes: int = 60) -> int:
        """Get number of recent upload attempts from an IP."""
        if ip_address not in self.upload_attempts:
            return 0

        cutoff = datetime.now().timestamp() - (minutes * 60)
        recent_attempts = [
            attempt
            for attempt in self.upload_attempts[ip_address]
            if attempt.timestamp() > cutoff
        ]
        return len(recent_attempts)

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        total_docs = len(self.documents_db)
        processing_docs = len(
            [
                doc
                for doc in self.documents_db.values()
                if doc["status"] == DocumentStatus.PROCESSING
            ]
        )
        completed_docs = len(
            [
                doc
                for doc in self.documents_db.values()
                if doc["status"] == DocumentStatus.DONE
            ]
        )
        failed_docs = len(
            [
                doc
                for doc in self.documents_db.values()
                if doc["status"] == DocumentStatus.FAILED
            ]
        )

        return {
            "total_documents": total_docs,
            "processing_documents": processing_docs,
            "completed_documents": completed_docs,
            "failed_documents": failed_docs,
            "processed_data_entries": len(self.processed_documents),
        }


# Global storage instance
document_storage = DocumentStorage()
