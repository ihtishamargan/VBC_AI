"""Simple authentication for VBC AI API."""

import hashlib
import os
import secrets
from datetime import datetime, timedelta

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# Simple in-memory token store (use Redis/database in production)
ACTIVE_TOKENS = {}
TOKEN_EXPIRY_HOURS = 24

# Simple API key authentication
API_KEYS = {
    "vbc_demo_key": {"name": "Demo User", "created_at": datetime.now()},
    "vbc_frontend_token_2024": {
        "name": "Frontend Client",
        "created_at": datetime.now(),
    },
}

# Add environment variable API keys if available
if os.getenv("VBC_API_KEY"):
    API_KEYS[os.getenv("VBC_API_KEY")] = {
        "name": "Environment User",
        "created_at": datetime.now(),
    }

if os.getenv("VBC_API_TOKEN"):
    API_KEYS[os.getenv("VBC_API_TOKEN")] = {
        "name": "Frontend Token",
        "created_at": datetime.now(),
    }


class AuthUser(BaseModel):
    """Authenticated user information."""

    name: str
    api_key: str


# Bearer token security scheme
security = HTTPBearer(auto_error=False)


def generate_token() -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


async def verify_api_key(x_api_key: str | None = Header(None)) -> AuthUser | None:
    """Verify API key from header."""
    if not x_api_key:
        return None

    if x_api_key in API_KEYS:
        key_info = API_KEYS[x_api_key]
        return AuthUser(name=key_info["name"], api_key=x_api_key)

    return None


async def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> AuthUser | None:
    """Verify bearer token."""
    if not credentials or not credentials.credentials:
        return None

    token = credentials.credentials
    if token in ACTIVE_TOKENS:
        token_info = ACTIVE_TOKENS[token]

        # Check if token is expired
        if datetime.now() > token_info["expires_at"]:
            del ACTIVE_TOKENS[token]
            return None

        return AuthUser(
            name=token_info["user_name"], api_key=token_info.get("api_key", "")
        )

    return None


async def get_current_user(
    api_user: AuthUser | None = Depends(verify_api_key),
    bearer_user: AuthUser | None = Depends(verify_bearer_token),
) -> AuthUser | None:
    """Get current authenticated user (API key or bearer token)."""
    return api_user or bearer_user


async def require_auth(
    current_user: AuthUser | None = Depends(get_current_user),
) -> AuthUser:
    """Require authentication - raises 401 if not authenticated."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide X-API-Key header or Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


# Simple auth - no permissions needed


def create_temp_token(api_key: str, duration_hours: int = TOKEN_EXPIRY_HOURS) -> str:
    """Create a temporary bearer token for an API key."""
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

    token = generate_token()
    key_info = API_KEYS[api_key]

    ACTIVE_TOKENS[token] = {
        "api_key": api_key,
        "user_name": key_info["name"],
        "created_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(hours=duration_hours),
    }

    return token


def revoke_token(token: str) -> bool:
    """Revoke a bearer token."""
    if token in ACTIVE_TOKENS:
        del ACTIVE_TOKENS[token]
        return True
    return False


def get_auth_stats() -> dict:
    """Get authentication statistics."""
    active_tokens = len(
        [t for t in ACTIVE_TOKENS.values() if datetime.now() <= t["expires_at"]]
    )

    return {
        "total_api_keys": len(API_KEYS),
        "active_tokens": active_tokens,
        "expired_tokens": len(ACTIVE_TOKENS) - active_tokens,
    }


def cleanup_expired_tokens():
    """Remove expired tokens from memory."""
    now = datetime.now()
    expired = [
        token for token, info in ACTIVE_TOKENS.items() if now > info["expires_at"]
    ]

    for token in expired:
        del ACTIVE_TOKENS[token]

    return len(expired)
