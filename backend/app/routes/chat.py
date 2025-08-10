"""Chat and search endpoints."""
import logging
from typing import Optional

from fastapi import APIRouter, Query, HTTPException

from backend.app.models import ChatRequest, ChatResponse, SearchResponse, Source
from backend.app.services import DocumentRetrievalService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/ai", tags=["ai"])

# Initialize services
document_retrieval = DocumentRetrievalService()

# Global counter for tracking queries
chat_queries_count = 0


@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest):
    """Chat with AI about documents with optional filters."""
    global chat_queries_count
    chat_queries_count += 1
    
    logger.info(f"Chat query: {request.message}")
    
    try:
        # Use the retrieval service for chat
        result = await document_retrieval.chat_with_documents(
            message=request.message,
            filters=request.filters,
            context_limit=5
        )
        
        return ChatResponse(
            answer=result["answer"],
            sources=[Source(**source) for source in result["sources"]]
        )
        
    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.get("/search", response_model=SearchResponse)
async def search_documents(
    q: str = Query(..., description="Search query"), 
    filters: Optional[str] = Query(None, description="JSON filters")
):
    """Search through document chunks with optional filters."""
    logger.info(f"Search query: {q}")
    
    try:
        # Parse filters if provided
        parsed_filters = None
        if filters:
            import json
            try:
                parsed_filters = json.loads(filters)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON filters: {filters}")
        
        # Use the retrieval service for search
        search_chunks = await document_retrieval.search_documents(
            query=q,
            filters=parsed_filters,
            limit=20
        )
        
        return SearchResponse(
            query=q,
            chunks=search_chunks,
            total_results=len(search_chunks)
        )
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


def get_chat_metrics():
    """Get chat-related metrics."""
    return {
        "total_queries": chat_queries_count
    }
