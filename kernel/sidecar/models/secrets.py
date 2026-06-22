"""G5 — secrets / OS-keychain for BYO keys.

Provides utility functions to safely retrieve and store API keys using the
system keyring (if available) with a transparent fallback to environment
variables.
"""
from __future__ import annotations

import os
from typing import Optional

_KEYRING_AVAILABLE = False
try:
    import keyring
    _KEYRING_AVAILABLE = True
except ImportError:
    pass

def get_api_key(service: str) -> Optional[str]:
    """Retrieve API key for service.

    Checks system keyring first under the service name 'kairo', then falls
    back to environment variables.
    """
    key: Optional[str] = None
    if _KEYRING_AVAILABLE:
        try:
            # Service name 'kairo', username is the service (e.g. 'openai')
            key = keyring.get_password("kairo", service)
        except Exception:
            pass

    if not key:
        # Fall back to env var e.g. OPENAI_API_KEY
        env_var_name = f"{service.upper()}_API_KEY"
        key = os.environ.get(env_var_name)

    return key

def set_api_key(service: str, key: str) -> None:
    """Store API key for service.

    Attempts to write to system keyring under service 'kairo'. If keyring is
    unavailable, does nothing. Never logs or prints the key.
    """
    if _KEYRING_AVAILABLE:
        try:
            keyring.set_password("kairo", service, key)
        except Exception:
            pass

def clear_api_key(service: str) -> None:
    """Clear API key for service from system keyring.

    If keyring is unavailable or key does not exist, does nothing.
    """
    if _KEYRING_AVAILABLE:
        try:
            keyring.delete_password("kairo", service)
        except Exception:
            pass

def redact_key(key: Optional[str]) -> str:
    """Return a redacted string representation of the key."""
    if not key:
        return "None"
    if len(key) <= 8:
        return "[REDACTED]"
    return f"{key[:4]}...{key[-4:]}"
