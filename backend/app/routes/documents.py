"""Document upload and processing endpoints."""
import logging
import uuid
import asyncio
import os
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks

from app.config import settings
from app.models import (
    DocumentUploadResponse, DocumentStatus, DocumentStatusResponse,
    ExtractedDocument, VBCContractData
)
from app.services import DocumentParser, DocumentIngestionService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/documents", tags=["documents"])

# In-memory storage (replace with actual database in production)
documents_db: Dict[str, Any] = {}
processed_documents: Dict[str, Any] = {}

# Initialize services
document_parser = DocumentParser(max_file_size_mb=settings.max_file_size_mb)
document_ingestion = DocumentIngestionService()

# Ensure upload directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.processed_dir, exist_ok=True)


async def process_document_background(document_id: str, file_path: str):
    """Background task to process uploaded document."""
    try:
        logger.info(f"Starting background processing for document {document_id}")
        
        # Update status to processing
        if document_id in documents_db:
            documents_db[document_id]["status"] = DocumentStatus.PROCESSING
            documents_db[document_id]["updated_at"] = datetime.now()
        
        # Step 1: Parse the document
        pages = await document_parser.parse_pdf(file_path)
        logger.info(f"Parsed {len(pages)} pages from document {document_id}")
        
        # Step 2: Ingest document (analyze, chunk, store vectors)
        result = await document_ingestion.ingest_document(pages, document_id)
        
        if result["success"]:
            # Store processed data
            processed_documents[document_id] = result
            
            # Update document status to done
            documents_db[document_id]["status"] = DocumentStatus.DONE
            documents_db[document_id]["updated_at"] = datetime.now()
            documents_db[document_id]["processing_time"] = result["processing_time"]
            
            logger.info(
                f"Document {document_id} processed successfully: "
                f"{result['chunks_created']} chunks, {result['vectors_stored']} vectors stored"
            )
        else:
            # Update status to error
            documents_db[document_id]["status"] = DocumentStatus.ERROR
            documents_db[document_id]["updated_at"] = datetime.now()
            documents_db[document_id]["error_message"] = result["error"]
            
            logger.error(f"Document {document_id} processing failed: {result['error']}")
            
    except Exception as e:
        logger.error(f"Background processing failed for document {document_id}: {str(e)}")
        
        # Update status to error
        if document_id in documents_db:
            documents_db[document_id]["status"] = DocumentStatus.ERROR
            documents_db[document_id]["updated_at"] = datetime.now()
            documents_db[document_id]["error_message"] = str(e)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and immediately process a document, returning complete analysis."""
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Check file size
    if file.size and file.size > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
        )
    
    # Generate document ID and save file
    document_id = str(uuid.uuid4())
    file_path = os.path.join(settings.upload_dir, f"{document_id}_{file.filename}")
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Saved uploaded file: {file.filename} -> {file_path}")
        
        # Store document metadata
        documents_db[document_id] = {
            "id": document_id,
            "filename": file.filename,
            "upload_time": datetime.now(),
            "status": DocumentStatus.PENDING,
            "updated_at": datetime.now(),
            "file_path": file_path,
            "file_size": len(content)
        }
        
        # **IMMEDIATE PROCESSING (not background) for complete response**
        logger.info(f"Starting immediate processing for document {document_id}")
        
        # Update status to processing
        documents_db[document_id]["status"] = DocumentStatus.PROCESSING
        documents_db[document_id]["updated_at"] = datetime.now()
        
        # Step 1: Parse the document
        pages = await document_parser.parse_pdf(file_path)
        logger.info(f"Parsed {len(pages)} pages from document {document_id}")
        
        # Step 2: Ingest document (analyze, chunk, store vectors)
        result = await document_ingestion.ingest_document(pages, document_id)
        
        if result["success"]:
            # Store processed data
            processed_documents[document_id] = result
            
            # Update document status to done
            documents_db[document_id]["status"] = DocumentStatus.DONE
            documents_db[document_id]["updated_at"] = datetime.now()
            documents_db[document_id]["processing_time"] = result["processing_time"]
            
            logger.info(
                f"Document {document_id} processed successfully: "
                f"{result['chunks_created']} chunks, {result['vectors_stored']} vectors stored"
            )
            
            # Prepare VBC analysis response
            vbc_analysis = None
            if "vbc_analysis" in result and isinstance(result["vbc_analysis"], dict):
                try:
                    vbc_analysis = VBCContractData(**result["vbc_analysis"])
                except Exception as e:
                    logger.warning(f"Failed to parse VBC analysis as structured data: {e}")
                    vbc_analysis = None
            
            # Return complete analysis results
            return DocumentUploadResponse(
                document_id=document_id,
                filename=file.filename,
                status=DocumentStatus.DONE,
                processing_time_seconds=result["processing_time"],
                file_size_bytes=len(content),
                pages_processed=result["pages_processed"],
                chunks_created=result["chunks_created"],
                vectors_stored=result["vectors_stored"],
                analysis_summary=result.get("analysis_summary", "Document processed successfully"),
                key_topics=result.get("topics", []),
                entities_found=result.get("entities", []),
                vbc_contract_data=vbc_analysis
            )
        
        else:
            # Processing failed
            documents_db[document_id]["status"] = DocumentStatus.ERROR
            documents_db[document_id]["updated_at"] = datetime.now()
            documents_db[document_id]["error_message"] = result.get("error", "Processing failed")
            
            logger.error(f"Document {document_id} processing failed: {result.get('error')}")
            
            # Return error response
            return DocumentUploadResponse(
                document_id=document_id,
                filename=file.filename,
                status=DocumentStatus.ERROR,
                processing_time_seconds=0.0,
                file_size_bytes=len(content),
                pages_processed=0,
                chunks_created=0,
                vectors_stored=0,
                error_message=result.get("error", "Processing failed")
            )
            
    except Exception as e:
        logger.error(f"Upload processing failed: {str(e)}")
        
        # Clean up file if it was saved
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Clean up document record
        if document_id in documents_db:
            del documents_db[document_id]
            
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")


@router.get("/status/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(document_id: str):
    """Get the processing status of a document."""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents_db[document_id]
    
    return DocumentStatusResponse(
        document_id=document_id,
        status=doc["status"],
        created_at=doc.get("upload_time", doc.get("created_at")),
        updated_at=doc["updated_at"],
        error_message=doc.get("error_message")
    )


@router.get("/extract/{document_id}", response_model=ExtractedDocument)
async def get_extracted_document(document_id: str):
    """Get the extracted and normalized content of a processed document."""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = documents_db[document_id]
    if doc["status"] != DocumentStatus.DONE:
        raise HTTPException(status_code=400, detail="Document not yet processed")
    
    if document_id not in processed_documents:
        raise HTTPException(status_code=404, detail="Processed document data not found")
    
    processed_data = processed_documents[document_id]
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
            "file_size": metadata.get("file_size", doc.get("file_size", 0))
        },
        "chunks_preview": processed_data.get("chunks", [])[:3],  # First 3 chunks as preview
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
