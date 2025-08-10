"""API routes package."""

from .chat import router as chat_router
from .documents import router as documents_router
from .system import router as system_router

__all__ = ["documents_router", "chat_router", "system_router"]
