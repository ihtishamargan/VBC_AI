"""Data models for document ingestion pipeline."""

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


class AnalysisStrategy(Enum):
    """Strategy for document analysis."""

    VBC_CONTRACT = "vbc_contract"
    GENERIC_LLM = "generic_llm"
    NONE = "none"


class IngestionStatus(Enum):
    """Status of document ingestion process."""

    PENDING = "pending"
    ANALYZING = "analyzing"
    CHUNKING = "chunking"
    VECTORIZING = "vectorizing"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata."""

    content: str
    metadata: dict[str, Any]
    chunk_id: str = None
    created_at: datetime = None

    def __post_init__(self):
        """Initialize auto-generated fields."""
        if self.chunk_id is None:
            self.chunk_id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class DeduplicationInfo(BaseModel):
    """Information about document deduplication."""

    checked: bool = False
    is_duplicate: bool = False
    duplicate_type: str | None = None
    was_update: bool = False
    file_hash: str | None = None
    content_hash: str | None = None


class IngestionResult(BaseModel):
    """Result of document ingestion process."""

    success: bool
    document_id: str
    status: IngestionStatus
    processing_time: float

    # Processing metrics
    pages_processed: int = 0
    chunks_created: int = 0
    vectors_stored: int = 0

    # Analysis results
    analysis: dict[str, Any] | None = None
    vbc_analysis: dict[str, Any] | None = None
    vbc_contract_id: str | None = None

    # Data
    chunks: list[dict[str, Any]] = []
    vector_ids: list[str] = []

    # Deduplication
    deduplication_info: DeduplicationInfo | None = None

    # Error information
    error: str | None = None
    error_details: dict[str, Any] | None = None

    @classmethod
    def success_result(
        cls,
        document_id: str,
        processing_time: float,
        pages_processed: int,
        chunks_created: int,
        vectors_stored: int,
        **kwargs,
    ) -> "IngestionResult":
        """Create a successful ingestion result."""
        return cls(
            success=True,
            document_id=document_id,
            status=IngestionStatus.COMPLETED,
            processing_time=processing_time,
            pages_processed=pages_processed,
            chunks_created=chunks_created,
            vectors_stored=vectors_stored,
            **kwargs,
        )

    @classmethod
    def failure_result(
        cls,
        document_id: str,
        processing_time: float,
        error: str,
        status: IngestionStatus = IngestionStatus.FAILED,
        **kwargs,
    ) -> "IngestionResult":
        """Create a failed ingestion result."""
        return cls(
            success=False,
            document_id=document_id,
            status=status,
            processing_time=processing_time,
            error=error,
            **kwargs,
        )


class IngestionConfig(BaseModel):
    """Configuration for document ingestion."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    analysis_strategy: AnalysisStrategy = AnalysisStrategy.VBC_CONTRACT
    llm_model: str = "gpt-4"
    collection_name: str = "vbc_documents"
    enable_deduplication: bool = True
    enable_database_storage: bool = True

    class Config:
        use_enum_values = True
