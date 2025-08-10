"""File validation utilities for document processing."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_pdf_content(content: bytes) -> bool:
    """Validate PDF content for basic structure and safety."""
    try:
        # Check minimum PDF size
        if len(content) < 100:
            return False

        # Check PDF header
        if not content.startswith(b"%PDF-"):
            return False

        # Check for PDF trailer
        if b"%%EOF" not in content[-500:]:  # Look for EOF in last 500 bytes
            logger.warning("PDF missing proper EOF marker")

        # Basic safety checks - reject files with suspicious content
        suspicious_patterns = [
            b"/JavaScript",  # Embedded JavaScript
            b"/JS",  # JavaScript action
            b"/EmbeddedFile",  # Embedded files
            b"/Launch",  # Launch actions
        ]

        for pattern in suspicious_patterns:
            if pattern in content:
                logger.warning(
                    f"PDF contains potentially unsafe content: {pattern.decode('utf-8', errors='ignore')}"
                )
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
    sanitized = "".join(c for c in safe_name if c in safe_chars)

    # Ensure it ends with .pdf
    if not sanitized.lower().endswith(".pdf"):
        sanitized += ".pdf"

    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:96] + ".pdf"

    return sanitized or "document.pdf"
