import logging
import uuid
import asyncio
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

from document_handler import DocumentHandler, DocumentMetadata, DocumentProcessingResult

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
processed_documents = {}  # Store processed PDF data
chat_queries_count = 0
app_start_time = datetime.now()

# Initialize document handler with modular pipeline
document_handler = DocumentHandler(
    chunk_size=1200,
    chunk_overlap=200,
    llm_model="gpt-4o-mini",
    collection_name="pdf_documents",
    enable_vector_store=True
)



async def process_document_background(document_id: str, file_path: str):
    """Background task to process uploaded document."""
    try:
        logger.info(f"Starting background processing for document {document_id}")
        
        # Update status to processing
        if document_id in documents_db:
            documents_db[document_id]["status"] = DocumentStatus.PROCESSING
            documents_db[document_id]["updated_at"] = datetime.now()
        
        # Process the document using modular handler
        result = await document_handler.process_document(file_path, document_id)
        
        if result.success:
            # Store processed data (convert to dict for consistency)
            processed_documents[document_id] = {
                "success": True,
                "metadata": result.metadata.model_dump(),
                "analysis": result.analysis.model_dump() if result.analysis else None,
                "chunks": [chunk.model_dump() for chunk in result.chunks],
                "vector_ids": result.vector_ids,
            }
            
            # Update document status to done
            documents_db[document_id]["status"] = DocumentStatus.DONE
            documents_db[document_id]["updated_at"] = datetime.now()
            documents_db[document_id]["processing_time"] = result.metadata.processing_time
            
            logger.info(f"Document {document_id} processed successfully: {len(result.chunks)} chunks, {len(result.vector_ids)} vectors stored")
        else:
            # Update status to error
            documents_db[document_id]["status"] = DocumentStatus.ERROR
            documents_db[document_id]["error_message"] = result.error
            documents_db[document_id]["updated_at"] = datetime.now()
            logger.error(f"Document {document_id} processing failed: {result.error}")
            
    except Exception as e:
        logger.error(f"Background processing failed for {document_id}: {str(e)}")
        if document_id in documents_db:
            documents_db[document_id]["status"] = DocumentStatus.ERROR
            documents_db[document_id]["error_message"] = str(e)
            documents_db[document_id]["updated_at"] = datetime.now()

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """Upload a document for processing."""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Read and save file
        file_content = await file.read()
        file_path = DocumentHandler.save_uploaded_file(file_content, file.filename)
        
        # Store document info
        documents_db[document_id] = {
            "id": document_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "status": DocumentStatus.QUEUED,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "file_size": len(file_content),
            "file_path": file_path
        }
        
        # Start background processing
        background_tasks.add_task(process_document_background, document_id, file_path)
        
        logger.info(f"Document uploaded: {document_id}, filename: {file.filename}, size: {len(file_content)} bytes")
        
        return DocumentUploadResponse(
            document_id=document_id,
            status=DocumentStatus.QUEUED,
            uploaded_at=datetime.now()
        )
        
    except HTTPException:
        raise
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
    
    # Check if we have processed data
    if document_id not in processed_documents:
        raise HTTPException(status_code=404, detail="Processed document data not found")
    
    processed_data = processed_documents[document_id]
    insights = processed_data["insights"]
    metadata = processed_data["metadata"]
    
    # Structure the extracted content using real processed data
    extracted_content = {
        "document_type": insights["document_type"],
        "summary": insights["summary"],
        "key_topics": insights["key_topics"],
        "entities": insights["entities"],
        "confidence_score": insights["confidence_score"],
        "metadata": {
            "total_pages": metadata["total_pages"],
            "total_chunks": metadata["total_chunks"],
            "processing_time": metadata["processing_time"],
            "file_size": metadata["file_size"]
        },
        "chunks_preview": processed_data["chunks"][:3],  # First 3 chunks as preview
        "raw_pages_preview": processed_data.get("raw_pages", [])
    }
    
    # Check for potential PHI based on entities and content
    redacted_fields = []
    for entity in insights.get("entities", []):
        if entity.get("type") in ["PERSON", "EMAIL", "PHONE", "SSN"]:
            redacted_fields.append(entity["type"].lower())
    
    # Additional PHI check based on filename
    if "personal" in doc.get("filename", "").lower():
        redacted_fields.extend(["personal_info", "contact_details"])
    
    return ExtractedDocument(
        document_id=document_id,
        content=extracted_content,
        extracted_at=datetime.fromisoformat(metadata["processed_at"]) if isinstance(metadata["processed_at"], str) else metadata["processed_at"],
        redacted_fields=list(set(redacted_fields))  # Remove duplicates
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
