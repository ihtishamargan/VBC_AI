"""Document deduplication service for VBC AI."""

import hashlib
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class DocumentDeduplicationService:
    """Handles document deduplication using multiple strategies."""

    def __init__(self):
        # In-memory storage for this session (replace with database in production)
        self.document_hashes: dict[str, dict[str, Any]] = {}
        self.agreement_ids: dict[str, dict[str, Any]] = {}

    def generate_file_hash(self, file_content: bytes) -> str:
        """Generate SHA-256 hash of file content."""
        return hashlib.sha256(file_content).hexdigest()

    def generate_content_hash(self, text_content: str) -> str:
        """Generate SHA-256 hash of text content (for content-based deduplication)."""
        # Normalize text: remove extra whitespace, convert to lowercase
        normalized = " ".join(text_content.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def check_file_hash_duplicate(
        self, file_hash: str
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Check if a document with this file hash already exists.

        Returns:
            (is_duplicate, existing_document_info)
        """
        if file_hash in self.document_hashes:
            existing_doc = self.document_hashes[file_hash]
            logger.info(f"Found duplicate by file hash: {file_hash[:16]}...")
            return True, existing_doc

        return False, None

    def check_agreement_duplicate(
        self, agreement_id: str
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Check if a document with this agreement ID already exists.

        Returns:
            (is_duplicate, existing_document_info)
        """
        if agreement_id and agreement_id.strip():
            normalized_agreement_id = agreement_id.strip().upper()
            if normalized_agreement_id in self.agreement_ids:
                existing_doc = self.agreement_ids[normalized_agreement_id]
                logger.info(f"Found duplicate by agreement ID: {agreement_id}")
                return True, existing_doc

        return False, None

    def check_content_similarity(
        self, content_hash: str, similarity_threshold: float = 0.95  # noqa: ARG002
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Check if similar content already exists.

        For now, this does exact content hash matching.
        Could be extended with semantic similarity using embeddings.
        """
        # Simple implementation: exact content hash match
        for _file_hash, doc_info in self.document_hashes.items():
            if doc_info.get("content_hash") == content_hash:
                logger.info(f"Found duplicate by content hash: {content_hash[:16]}...")
                return True, doc_info

        return False, None

    def register_document(
        self,
        document_id: str,
        filename: str,
        file_hash: str,
        content_hash: str,
        agreement_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Register a new document in the deduplication system.

        Args:
            document_id: Unique document identifier
            filename: Original filename
            file_hash: SHA-256 hash of file content
            content_hash: SHA-256 hash of text content
            agreement_id: VBC agreement identifier (if available)
            metadata: Additional document metadata

        Returns:
            Document info dictionary
        """
        doc_info = {
            "document_id": document_id,
            "filename": filename,
            "file_hash": file_hash,
            "content_hash": content_hash,
            "agreement_id": agreement_id,
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # Register by file hash
        self.document_hashes[file_hash] = doc_info

        # Register by agreement ID if available
        if agreement_id and agreement_id.strip():
            normalized_agreement_id = agreement_id.strip().upper()
            self.agreement_ids[normalized_agreement_id] = doc_info
            logger.info(f"Registered document by agreement ID: {agreement_id}")

        logger.info(f"Registered document: {document_id} (hash: {file_hash[:16]}...)")
        return doc_info

    def update_document(
        self,
        existing_doc_info: dict[str, Any],
        new_filename: str,
        new_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Update existing document information.

        Args:
            existing_doc_info: Existing document information
            new_filename: New filename (in case of rename)
            new_metadata: Updated metadata

        Returns:
            Updated document info
        """
        # Update the existing document info
        existing_doc_info["filename"] = new_filename
        existing_doc_info["updated_at"] = datetime.now().isoformat()

        if new_metadata:
            existing_doc_info["metadata"].update(new_metadata)

        logger.info(f"Updated document: {existing_doc_info['document_id']}")
        return existing_doc_info

    def perform_comprehensive_check(
        self,
        file_content: bytes,
        text_content: str,
        filename: str,  # noqa: ARG002
        agreement_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Perform comprehensive deduplication check using multiple strategies.

        Returns:
            {
                'is_duplicate': bool,
                'duplicate_type': str,  # 'file_hash', 'agreement_id', 'content_hash'
                'existing_document': dict or None,
                'file_hash': str,
                'content_hash': str,
                'should_update': bool  # Whether to update existing vs skip
            }
        """
        file_hash = self.generate_file_hash(file_content)
        content_hash = self.generate_content_hash(text_content)

        result = {
            "is_duplicate": False,
            "duplicate_type": None,
            "existing_document": None,
            "file_hash": file_hash,
            "content_hash": content_hash,
            "should_update": False,
        }

        # 1. Check file hash (exact file duplicate)
        is_file_duplicate, existing_doc = self.check_file_hash_duplicate(file_hash)
        if is_file_duplicate:
            result.update(
                {
                    "is_duplicate": True,
                    "duplicate_type": "file_hash",
                    "existing_document": existing_doc,
                    "should_update": False,  # Exact same file, no need to update
                }
            )
            return result

        # 2. Check agreement ID (same contract, potentially different file)
        if agreement_id:
            is_agreement_duplicate, existing_doc = self.check_agreement_duplicate(
                agreement_id
            )
            if is_agreement_duplicate:
                result.update(
                    {
                        "is_duplicate": True,
                        "duplicate_type": "agreement_id",
                        "existing_document": existing_doc,
                        "should_update": True,  # Same agreement, but different file - update
                    }
                )
                return result

        # 3. Check content similarity (same content, different filename/format)
        is_content_duplicate, existing_doc = self.check_content_similarity(content_hash)
        if is_content_duplicate:
            result.update(
                {
                    "is_duplicate": True,
                    "duplicate_type": "content_hash",
                    "existing_document": existing_doc,
                    "should_update": False,  # Same content, likely just renamed
                }
            )
            return result

        # No duplicates found
        return result

    def get_stats(self) -> dict[str, Any]:
        """Get deduplication statistics."""
        return {
            "total_documents": len(self.document_hashes),
            "documents_by_agreement": len(self.agreement_ids),
            "file_hashes": list(self.document_hashes.keys()),
            "agreement_ids": list(self.agreement_ids.keys()),
        }


# Global instance (in production, this would be a proper service with database)
deduplication_service = DocumentDeduplicationService()
