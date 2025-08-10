"""System health and metrics endpoints."""
import logging
from datetime import datetime

from fastapi import APIRouter

from backend.app.models import HealthResponse, MetricsResponse
from backend.app.routes.documents import documents_db
from backend.app.routes.chat import get_chat_metrics
from backend.app.models import DocumentStatus

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["system"])

# Track application start time
app_start_time = datetime.now()


@router.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now()
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get application metrics."""
    
    uptime = (datetime.now() - app_start_time).total_seconds()
    processing_docs = len([doc for doc in documents_db.values() if doc["status"] == DocumentStatus.PROCESSING])
    chat_metrics = get_chat_metrics()
    
    return MetricsResponse(
        total_documents=len(documents_db),
        processing_documents=processing_docs,
        total_queries=chat_metrics["total_queries"],
        uptime_seconds=uptime
    )


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "VBC AI API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/healthz", 
        "metrics": "/metrics"
    }
