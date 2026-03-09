from __future__ import annotations

import os
import re
from enum import Enum
from typing import Dict, List

# WrapperServiceError: try to import from project, fallback to a local stub
try:
    from wrapper_service.exceptions import WrapperServiceError  # type: ignore
except Exception:

    class WrapperServiceError(Exception):
        pass


class AuthError(WrapperServiceError):
    pass


class APIKeyAuth:
    """API Key based authentication.

    api_keys_config: Dict[str, List[str]]
        Keys mapped to a list of permissions, e.g. {"test_key": ["read", "write"]}
    """

    def __init__(self, api_keys_config: Dict[str, List[str]]):
        self._keys: Dict[str, List[str]] = api_keys_config or {}

    def authenticate(self, api_key: str) -> List[str]:
        if not api_key:
            raise AuthError("API key is missing")
        perms = self._keys.get(api_key)
        if perms is None:
            raise AuthError("Invalid API key")
        return perms

    def check_permission(self, permissions: List[str], required: str) -> bool:
        if permissions is None:
            return False
        return required in permissions


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


def _parse_env_keys() -> Dict[str, List[str]]:
    raw = os.getenv("WRAPPER_API_KEYS")
    if not raw:
        return {}
    result: Dict[str, List[str]] = {}
    # Entries separated by comma: key1:read, key2:read;write, key3:admin
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        key, perms_str = part.split(":", 1)
        key = key.strip()
        perms = [p.strip() for p in re.split(r"[;\,\s]+", perms_str) if p.strip()]
        perms = [p for p in perms if p in {"read", "write", "admin"}]
        if key and perms:
            result[key] = perms
    return result


# Global cached API keys configuration loaded from environment
_API_KEYS_CONFIG: Dict[str, List[str]] = _parse_env_keys()
_API_KEY_AUTH = APIKeyAuth(_API_KEYS_CONFIG)


# FastAPI dependencies (optional import-safe)
try:
    from fastapi import Header, Depends, HTTPException  # type: ignore

    try:
        from wrapper_service.config import Settings, get_settings  # type: ignore
    except Exception:

        class Settings:  # type: ignore
            auth_enabled: bool = True

        async def get_settings():  # type: ignore
            return Settings()
except Exception:
    # Fallbacks for environments without FastAPI at import time (e.g., tests)
    Header = lambda *args, **kwargs: None  # type: ignore
    Depends = lambda dep=None: dep  # type: ignore
    HTTPException = Exception  # type: ignore

    class Settings:  # type: ignore
        auth_enabled = True

    async def get_settings():  # type: ignore
        return Settings()


async def require_auth(
    x_api_key: str = Header(..., description="API Key"),
    settings: "Settings" = Depends(get_settings),
) -> List[str]:
    # If authentication is disabled, allow public access
    if getattr(settings, "auth_enabled", True) is False:
        return []
    try:
        perms = _API_KEY_AUTH.authenticate(x_api_key)
        return perms
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc))


async def require_permission(required: Permission):
    # Returns a FastAPI dependency that enforces the given permission
    from fastapi import HTTPException  # local import to avoid hard dependency at import time

    async def _dep(permissions: List[str] = Depends(require_auth)) -> bool:
        if _API_KEY_AUTH.check_permission(permissions, required.value):
            return True
        raise HTTPException(status_code=403, detail="Forbidden")

    return _dep
