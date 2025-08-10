"""Document upload and processing endpoints."""
import logging
import uuid
import asyncio
import os
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
import mimetypes

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, status

from backend.app.config import settings
from backend.app.models import (
    DocumentUploadResponse, DocumentStatus, DocumentStatusResponse,
    ExtractedDocument, VBCContractData
)
from backend.app.services import DocumentParser, DocumentIngestionService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/documents", tags=["documents"])

# In-memory storage (replace with actual database in production)
documents_db: Dict[str, Any] = {}
processed_documents: Dict[str, Any] = {}

# Simple rate limiting (in production, use Redis or proper rate limiter)
upload_attempts: Dict[str, list] = {}  # IP -> list of timestamps

# Initialize services
document_parser = DocumentParser(max_file_size_mb=settings.max_file_size_mb)
document_ingestion = DocumentIngestionService()

# Ensure upload directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.processed_dir, exist_ok=True)


def validate_pdf_content(content: bytes) -> bool:
    """Validate PDF content for basic structure and safety."""
    try:
        # Check minimum PDF size
        if len(content) < 100:
            return False
        
        # Check PDF header
        if not content.startswith(b'%PDF-'):
            return False
        
        # Check for PDF trailer
        if b'%%EOF' not in content[-500:]:  # Look for EOF in last 500 bytes
            logger.warning("PDF missing proper EOF marker")
        
        # Basic safety checks - reject files with suspicious content
        suspicious_patterns = [
            b'/JavaScript',  # Embedded JavaScript
            b'/JS',          # JavaScript action
            b'/EmbeddedFile', # Embedded files
            b'/Launch',      # Launch actions
        ]
        
        for pattern in suspicious_patterns:
            if pattern in content:
                logger.warning(f"PDF contains potentially unsafe content: {pattern.decode('utf-8', errors='ignore')}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"PDF validation error: {e}")
        return False


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent security issues."""
    if not filename:
        return "document.pdf"
    
    # Remove path components
    safe_name = Path(filename).name
    
    # Remove potentially dangerous characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    sanitized = ''.join(c for c in safe_name if c in safe_chars)
    
    # Ensure it ends with .pdf
    if not sanitized.lower().endswith('.pdf'):
        sanitized += '.pdf'
    
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:96] + '.pdf'
    
    return sanitized or "document.pdf"


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
    
    # Input validation and sanitization
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No file provided or filename is empty"
        )
    
    # Sanitize filename to prevent path traversal attacks
    original_filename = file.filename
    safe_filename = sanitize_filename(original_filename)
    if safe_filename != original_filename:
        logger.info(f"Filename sanitized: {original_filename} -> {safe_filename}")
    
    # Validate file extension
    if not safe_filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Only PDF files are supported. Please upload a valid PDF document."
        )
    
    # Additional MIME type validation
    mime_type, _ = mimetypes.guess_type(safe_filename)
    if mime_type != 'application/pdf':
        logger.warning(f"File {safe_filename} has unexpected MIME type: {mime_type}")
    
    # Check file size
    if file.size:
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        if file.size > max_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large ({file.size / 1024 / 1024:.1f}MB). Maximum allowed: {settings.max_file_size_mb}MB"
            )
    
    # Validate filename length
    if len(safe_filename) > 255:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename too long. Maximum 255 characters allowed."
        )
    
    # Generate document ID and create secure file path
    document_id = str(uuid.uuid4())
    secure_filename = f"{document_id}_{safe_filename}"
    file_path = os.path.join(settings.upload_dir, secure_filename)
    
    # Ensure file path is within upload directory (prevent directory traversal)
    if not os.path.abspath(file_path).startswith(os.path.abspath(settings.upload_dir)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path"
        )
    
    temp_file_path = None
    try:
        # Read file content with size validation
        logger.info(f"Processing upload: {safe_filename} ({file.size} bytes)")
        content = await file.read()
        
        # Double-check actual content size
        if len(content) > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Actual file size ({len(content) / 1024 / 1024:.1f}MB) exceeds limit"
            )
        
        # Advanced PDF content validation
        if not validate_pdf_content(content):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or potentially unsafe PDF file. Please upload a clean, standard PDF document."
            )
        
        # Save to temporary location first for atomic operations
        temp_file_path = file_path + ".tmp"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)
        
        # Move to final location atomically
        os.rename(temp_file_path, file_path)
        temp_file_path = None  # Successfully moved
        
        logger.info(f"Saved uploaded file: {safe_filename} -> {file_path}")
        
        # Store document metadata with sanitized data
        documents_db[document_id] = {
            "id": document_id,
            "filename": safe_filename,  # Store sanitized filename
            "original_filename": original_filename,  # Keep track of original
            "upload_time": datetime.now(),
            "status": DocumentStatus.QUEUED,
            "updated_at": datetime.now(),
            "file_path": file_path,
            "file_size": len(content),
            "content_type": "application/pdf",
            "is_validated": True
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
            
            # Return complete analysis results with safe data
            return DocumentUploadResponse(
                document_id=document_id,
                filename=safe_filename,  # Return sanitized filename
                status=DocumentStatus.DONE,
                processing_time_seconds=result["processing_time"],
                file_size_bytes=len(content),
                pages_processed=result["pages_processed"],
                chunks_created=result["chunks_created"],
                vectors_stored=result["vectors_stored"],
                analysis_summary=result.get("analysis_summary", "Document processed successfully"),
                key_topics=result.get("topics", [])[:10],  # Limit response size
                entities_found=result.get("entities", [])[:20],  # Limit response size
                vbc_contract_data=vbc_analysis
            )
        
        else:
            # Processing failed
            documents_db[document_id]["status"] = DocumentStatus.ERROR
            documents_db[document_id]["updated_at"] = datetime.now()
            documents_db[document_id]["error_message"] = result.get("error", "Processing failed")
            
            logger.error(f"Document {document_id} processing failed: {result.get('error')}")
            
            # Return error response with safe data
            return DocumentUploadResponse(
                document_id=document_id,
                filename=safe_filename,  # Return sanitized filename
                status=DocumentStatus.ERROR,
                processing_time_seconds=0.0,
                file_size_bytes=len(content),
                pages_processed=0,
                chunks_created=0,
                vectors_stored=0,
                error_message=result.get("error", "Processing failed")
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Upload processing failed for {safe_filename}: {str(e)}", exc_info=True)
        
        # Clean up temporary file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
        
        # Clean up final file if it was saved
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up file {file_path}: {cleanup_error}")
        
        # Clean up document record
        if 'document_id' in locals() and document_id in documents_db:
            try:
                del documents_db[document_id]
            except Exception:
                pass
        
        # Return generic error message to avoid information disclosure
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Document processing failed. Please try again or contact support."
        )


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
