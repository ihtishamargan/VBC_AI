import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VBC AI API",
    description="Document processing and AI chat API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enums
class DocumentStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"

# Pydantic Models
class DocumentUploadResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    uploaded_at: datetime

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

class ChatRequest(BaseModel):
    message: str
    filters: Optional[Dict[str, Any]] = None

class Source(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]

class SearchRequest(BaseModel):
    q: str
    filters: Optional[Dict[str, Any]] = None

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

# In-memory storage (replace with actual database in production)
documents_db = {}
chat_queries_count = 0
app_start_time = datetime.now()

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for processing."""
    try:
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Store document info
        documents_db[document_id] = {
            "id": document_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "status": DocumentStatus.QUEUED,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "file_size": 0,  # Would be populated with actual file size
        }
        
        logger.info(f"Document uploaded: {document_id}, filename: {file.filename}")
        
        return DocumentUploadResponse(
            document_id=document_id,
            status=DocumentStatus.QUEUED,
            uploaded_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload document")

@app.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(document_id: str):
    """Get the processing status of a document."""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents_db[document_id]
    return DocumentStatusResponse(
        document_id=document_id,
        status=doc["status"],
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        error_message=doc.get("error_message")
    )

@app.get("/documents/{document_id}/extracted", response_model=ExtractedDocument)
async def get_extracted_document(document_id: str):
    """Get the extracted and normalized content of a processed document."""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents_db[document_id]
    
    if doc["status"] != DocumentStatus.DONE:
        raise HTTPException(
            status_code=422, 
            detail=f"Document is not ready. Current status: {doc['status']}"
        )
    
    # Mock extracted content (replace with actual extraction logic)
    extracted_content = {
        "contract_type": "Service Agreement",
        "parties": ["Company A", "Company B"],
        "effective_date": "2024-01-01",
        "terms": {
            "duration": "12 months",
            "payment_schedule": "Monthly"
        },
        "clauses": [
            {"section": "1", "title": "Services", "content": "Provider shall deliver..."},
            {"section": "2", "title": "Payment", "content": "Client shall pay..."}
        ]
    }
    
    # Mock PHI redaction
    redacted_fields = ["ssn", "phone_number", "email"] if "personal" in doc.get("filename", "").lower() else []
    
    return ExtractedDocument(
        document_id=document_id,
        content=extracted_content,
        extracted_at=datetime.now(),
        redacted_fields=redacted_fields
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with AI about documents with optional filters."""
    global chat_queries_count
    chat_queries_count += 1
    
    logger.info(f"Chat query: {request.message}")
    
    # Mock AI response (replace with actual AI logic)
    mock_sources = [
        Source(
            document_id="doc-123",
            chunk_id="chunk-456",
            content="This contract specifies payment terms of 30 days...",
            score=0.85,
            metadata={"page": 1, "section": "Payment Terms"}
        ),
        Source(
            document_id="doc-789",
            chunk_id="chunk-101",
            content="The service level agreement guarantees 99.9% uptime...",
            score=0.78,
            metadata={"page": 3, "section": "SLA"}
        )
    ]
    
    return ChatResponse(
        answer=f"Based on your question '{request.message}', I found relevant information in the uploaded contracts. The payment terms typically range from 30-60 days, and service level agreements commonly guarantee 99.9% uptime.",
        sources=mock_sources
    )

@app.get("/search", response_model=SearchResponse)
async def search(q: str = Query(..., description="Search query"), filters: Optional[str] = Query(None, description="JSON filters")):
    """Search through document chunks with optional filters."""
    logger.info(f"Search query: {q}")
    
    # Mock search results (replace with actual search logic)
    mock_chunks = [
        SearchChunk(
            chunk_id="chunk-001",
            document_id="doc-123",
            content=f"Content related to '{q}' found in contract section...",
            score=0.92,
            metadata={"page": 2, "document_type": "contract", "date": "2024-01-01"}
        ),
        SearchChunk(
            chunk_id="chunk-002",
            document_id="doc-456",
            content=f"Additional information about '{q}' in terms and conditions...",
            score=0.87,
            metadata={"page": 5, "document_type": "terms", "date": "2024-02-15"}
        )
    ]
    
    return SearchResponse(
        query=q,
        chunks=mock_chunks,
        total_results=len(mock_chunks)
    )

@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now()
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get application metrics."""
    uptime = (datetime.now() - app_start_time).total_seconds()
    processing_docs = len([doc for doc in documents_db.values() if doc["status"] == DocumentStatus.PROCESSING])
    
    return MetricsResponse(
        total_documents=len(documents_db),
        processing_documents=processing_docs,
        total_queries=chat_queries_count,
        uptime_seconds=uptime
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "VBC AI API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/healthz",
        "metrics": "/metrics"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
