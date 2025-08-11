"""Pydantic models for VBC AI RAG backend."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel
from dataclasses import dataclass

from backend.app.config import settings


# Enums
class DocumentStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


# Request Models
class ChatRequest(BaseModel):
    message: str
    filters: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    q: str
    filters: dict[str, Any] | None = None


# Response Models
class ProcessingMetrics(BaseModel):
    total_pages: int
    total_chunks: int
    vectors_stored: int
    processing_time: float
    file_size: int


class DocumentAnalysisResult(BaseModel):
    document_type: str
    summary: str
    key_topics: list[str]
    entities: list[dict[str, Any]]
    confidence_score: float
    processing_metrics: ProcessingMetrics
    chunks_preview: list[dict[str, Any]] = []


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    processing_time_seconds: float
    file_size_bytes: int
    pages_processed: int
    chunks_created: int
    vectors_stored: int
    analysis_summary: str | None = None
    key_topics: list[str] = []
    entities_found: list[dict[str, Any]] = []
    vbc_contract_data: Optional["VBCContractData"] = None
    error_message: str | None = None


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None


class ExtractedDocument(BaseModel):
    document_id: str
    content: dict[str, Any]
    extracted_at: datetime
    redacted_fields: list[str] = []


class Source(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = {}


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class SearchChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    chunks: list[SearchChunk]
    total_results: int


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime


class MetricsResponse(BaseModel):
    total_documents: int
    processing_documents: int
    total_queries: int
    uptime_seconds: float


# VBC Contract Models
class PaymentModelType(str, Enum):
    """Types of payment models in VBC contracts."""

    CAPITATION = "capitation"
    SHARED_SAVINGS = "shared_savings"
    RISK_SHARING = "risk_sharing"
    BUNDLED_PAYMENT = "bundled_payment"
    OUTCOME_BASED = "outcome_based"
    HYBRID = "hybrid"
    OTHER = "other"


class DiseaseAreaType(str, Enum):
    """Common disease areas in VBC contracts."""

    DIABETES = "diabetes"
    RHEUMATOID_ARTHRITIS = "rheumatoid_arthritis"
    HEART_FAILURE = "heart_failure"
    ONCOLOGY = "oncology"
    MENTAL_HEALTH = "mental_health"
    RESPIRATORY = "respiratory"
    CARDIOVASCULAR = "cardiovascular"
    IMMUNOLOGY = "immunology"
    RARE_DISEASE = "rare_disease"
    OTHER = "other"


class ContractParty(BaseModel):
    """Individual party in a VBC contract."""

    name: str
    type: str
    country: str | None = None


class FinancialStructure(BaseModel):
    """Financial structure and payment details."""

    initial_payment: float | None = None
    currency: str = "USD"
    payment_model: PaymentModelType
    base_reimbursement: float | None = None
    shared_savings_percentage: float | None = None
    risk_corridor_upper: float | None = None
    risk_corridor_lower: float | None = None
    maximum_payout: float | None = None
    minimum_guarantee: float | None = None


class RiskProtection(BaseModel):
    """Risk protection mechanisms."""

    has_stop_loss: bool = False
    has_risk_cap: bool = False
    has_non_responder_fund: bool = False
    stop_loss_threshold: float | None = None
    risk_cap_percentage: float | None = None


class OutcomeMetric(BaseModel):
    """Individual outcome metric definition."""

    name: str
    type: str
    target_value: str | None = None
    measurement_period: str | None = None
    data_source: str | None = None
    weight: float | None = None


class VBCContractData(BaseModel):
    """Complete structured data model for Value-Based Care contracts."""

    # Core identification
    agreement_id: str | None = None
    agreement_title: str

    # Parties and location
    parties: list[ContractParty]
    country: str

    # Clinical focus
    disease_area: DiseaseAreaType
    disease_area_details: str | None = None
    patient_population_size: int | None = None
    patient_population_description: str | None = None

    # Contract overview
    agreement_overview: str
    contract_background: str | None = None
    pilot_program_results: str | None = None

    # Financial terms
    financial_structure: FinancialStructure

    # Duration and timeline
    duration_months: int | None = None
    duration_description: str | None = None
    start_date: str | None = None
    end_date: str | None = None

    # Risk management
    risk_protection: RiskProtection = RiskProtection()

    # Outcome metrics
    outcome_metrics: list[OutcomeMetric] = []
    primary_endpoints: list[str] | None = None
    secondary_endpoints: list[str] | None = None

    # Quality and performance
    quality_measures: list[str] | None = None
    performance_benchmarks: list[str] | None = None

    # Data and reporting
    data_collection_frequency: str | None = None
    reporting_requirements: list[str] | None = None

    # Compliance and governance
    regulatory_framework: str | None = None
    governance_structure: str | None = None

    # Additional metadata
    contract_complexity: str | None = None
    innovation_elements: list[str] | None = None

    # Confidence and processing
    extraction_confidence: float = 0.0
    processing_notes: list[str] | None = None


class VBCContractAnalysisResponse(BaseModel):
    """Response model for VBC contract analysis."""

    success: bool
    contract_data: VBCContractData | None
    error_message: str | None
    processing_time_seconds: float
    pages_analyzed: int
    extraction_method: str = "llm_structured"

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
    pages_processed: int = 0
    chunks_created: int = 0
    vectors_stored: int = 0
    analysis: dict[str, Any] | None = None
    vbc_analysis: dict[str, Any] | None = None
    vbc_contract_id: str | None = None
    chunks: list[dict[str, Any]] = []
    vector_ids: list[str] = []
    deduplication_info: DeduplicationInfo | None = None
    error: str | None = None
    error_details: dict[str, Any] | None = None

    @classmethod
    def success_result(cls, document_id: str, processing_time: float, pages_processed: int, chunks_created: int, vectors_stored: int, **kwargs) -> "IngestionResult":
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
    def failure_result(cls, document_id: str, processing_time: float, error: str, status: IngestionStatus = IngestionStatus.FAILED, **kwargs) -> "IngestionResult":
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
    chunk_size: int = 4000
    chunk_overlap: int = 200
    analysis_strategy: AnalysisStrategy = AnalysisStrategy.VBC_CONTRACT
    llm_model: str = settings.openai_model
    collection_name: str = "contracts-index"
    enable_deduplication: bool = True
    enable_database_storage: bool = True

    class Config:
        use_enum_values = True
