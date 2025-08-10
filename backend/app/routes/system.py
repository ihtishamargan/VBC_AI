"""System health and metrics endpoints."""

from datetime import datetime

from fastapi import APIRouter

from backend.app.models import HealthResponse, MetricsResponse
from backend.app.services.chat_service import chat_service
from backend.app.storage.document_storage import document_storage
from backend.app.utils.logger import get_module_logger

# Configure logging
logger = get_module_logger(__name__)

# Create router
router = APIRouter(tags=["system"])

# Track application start time
app_start_time = datetime.now()


@router.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", timestamp=datetime.now())


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get application metrics."""

    uptime = (datetime.now() - app_start_time).total_seconds()
    storage_stats = document_storage.get_storage_stats()
    chat_metrics = chat_service.get_chat_metrics()

    return MetricsResponse(
        total_documents=storage_stats["total_documents"],
        processing_documents=storage_stats["processing_documents"],
        total_queries=chat_metrics["total_queries"],
        uptime_seconds=uptime,
    )


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "VBC AI API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/healthz",
        "metrics": "/metrics",
    }
