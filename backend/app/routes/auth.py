"""Authentication endpoints."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.app.auth import (
    API_KEYS,
    AuthUser,
    cleanup_expired_tokens,
    create_temp_token,
    get_auth_stats,
    require_auth,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["authentication"])


class TokenRequest(BaseModel):
    api_key: str
    duration_hours: int | None = 24


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user_name: str


class AuthStatus(BaseModel):
    authenticated: bool
    user_name: str
    api_key_prefix: str


@router.post("/token", response_model=TokenResponse)
async def create_access_token(request: TokenRequest):
    """Create a temporary bearer token from API key."""
    try:
        token = create_temp_token(request.api_key, request.duration_hours or 24)

        user_info = API_KEYS[request.api_key]

        return TokenResponse(
            access_token=token,
            token_type="bearer",
            expires_in=(request.duration_hours or 24) * 3600,  # Convert to seconds
            user_name=user_info["name"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token creation failed: {e}")
        raise HTTPException(status_code=500, detail="Token creation failed")


@router.delete("/token")
async def revoke_access_token(current_user: AuthUser = Depends(require_auth)):
    """Revoke the current bearer token."""
    # Note: This is a simplified implementation
    # In a real system, you'd extract the token from the Authorization header
    return {"message": "Token revocation requested (implementation simplified)"}


@router.get("/status", response_model=AuthStatus)
async def get_auth_status(current_user: AuthUser = Depends(require_auth)):
    """Get current authentication status."""
    return AuthStatus(
        authenticated=True,
        user_name=current_user.name,
        api_key_prefix=current_user.api_key[:8] + "..."
        if current_user.api_key
        else "token",
    )


@router.get("/stats")
async def get_authentication_stats(current_user: AuthUser = Depends(require_auth)):
    """Get authentication system statistics (requires auth)."""
    # Clean up expired tokens first
    expired_count = cleanup_expired_tokens()
    stats = get_auth_stats()

    return {
        **stats,
        "expired_tokens_cleaned": expired_count,
        "last_cleanup": datetime.now().isoformat(),
    }


@router.get("/health")
async def auth_health():
    """Public endpoint to check auth system health."""
    return {
        "status": "healthy",
        "auth_system": "active",
        "supported_methods": ["api_key", "bearer_token"],
        "timestamp": datetime.now().isoformat(),
    }
