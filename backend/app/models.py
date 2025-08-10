"""Pydantic models for VBC AI RAG backend."""
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


# Enums
class DocumentStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


# Request Models
class ChatRequest(BaseModel):
    message: str
    filters: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    q: str
    filters: Optional[Dict[str, Any]] = None


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
    key_topics: List[str]
    entities: List[Dict[str, Any]]
    confidence_score: float
    processing_metrics: ProcessingMetrics
    chunks_preview: List[Dict[str, Any]] = []


class DocumentUploadResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    uploaded_at: datetime
    filename: Optional[str] = None
    analysis: Optional[DocumentAnalysisResult] = None


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None


class ExtractedDocument(BaseModel):
    document_id: str
    content: Dict[str, Any]
    extracted_at: datetime
    redacted_fields: List[str] = []


class Source(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]


class SearchChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    chunks: List[SearchChunk]
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
    country: Optional[str] = None


class FinancialStructure(BaseModel):
    """Financial structure and payment details."""
    initial_payment: Optional[float] = None
    currency: str = "USD"
    payment_model: PaymentModelType
    base_reimbursement: Optional[float] = None
    shared_savings_percentage: Optional[float] = None
    risk_corridor_upper: Optional[float] = None
    risk_corridor_lower: Optional[float] = None
    maximum_payout: Optional[float] = None
    minimum_guarantee: Optional[float] = None


class RiskProtection(BaseModel):
    """Risk protection mechanisms."""
    has_stop_loss: bool = False
    has_risk_cap: bool = False
    has_non_responder_fund: bool = False
    stop_loss_threshold: Optional[float] = None
    risk_cap_percentage: Optional[float] = None


class OutcomeMetric(BaseModel):
    """Individual outcome metric definition."""
    name: str
    type: str
    target_value: Optional[str] = None
    measurement_period: Optional[str] = None
    data_source: Optional[str] = None
    weight: Optional[float] = None


class VBCContractData(BaseModel):
    """Complete structured data model for Value-Based Care contracts."""
    
    # Core identification
    agreement_id: Optional[str] = None
    agreement_title: str
    
    # Parties and location
    parties: List[ContractParty]
    country: str
    
    # Clinical focus
    disease_area: DiseaseAreaType
    disease_area_details: Optional[str] = None
    patient_population_size: Optional[int] = None
    patient_population_description: Optional[str] = None
    
    # Contract overview
    agreement_overview: str
    contract_background: Optional[str] = None
    pilot_program_results: Optional[str] = None
    
    # Financial terms
    financial_structure: FinancialStructure
    
    # Duration and timeline
    duration_months: Optional[int] = None
    duration_description: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Risk management
    risk_protection: RiskProtection = RiskProtection()
    
    # Outcome metrics
    outcome_metrics: List[OutcomeMetric] = []
    primary_endpoints: Optional[List[str]] = None
    secondary_endpoints: Optional[List[str]] = None
    
    # Quality and performance
    quality_measures: Optional[List[str]] = None
    performance_benchmarks: Optional[List[str]] = None
    
    # Data and reporting
    data_collection_frequency: Optional[str] = None
    reporting_requirements: Optional[List[str]] = None
    
    # Compliance and governance
    regulatory_framework: Optional[str] = None
    governance_structure: Optional[str] = None
    
    # Additional metadata
    contract_complexity: Optional[str] = None
    innovation_elements: Optional[List[str]] = None
    
    # Confidence and processing
    extraction_confidence: float = 0.0
    processing_notes: Optional[List[str]] = None


class VBCContractAnalysisResponse(BaseModel):
    """Response model for VBC contract analysis."""
    success: bool
    contract_data: Optional[VBCContractData]
    error_message: Optional[str]
    processing_time_seconds: float
    pages_analyzed: int
    extraction_method: str = "llm_structured"
