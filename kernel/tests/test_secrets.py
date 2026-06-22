"""Unit tests for kernel/sidecar/models/secrets.py.

Covers keyring retrieval/storage and fallback to environment variables.
"""
import sys
import os
import importlib
import pytest
from unittest.mock import MagicMock

import kernel.sidecar.models.secrets as secrets

def test_get_api_key_env_fallback(monkeypatch):
    """Verify that get_api_key falls back to env vars when keyring is not available."""
    # Temporarily set _KEYRING_AVAILABLE to False in the module
    monkeypatch.setattr(secrets, "_KEYRING_AVAILABLE", False)
    monkeypatch.setenv("OPENAI_API_KEY", "env")
    val = secrets.get_api_key("openai")
    assert val == "env"

def test_get_api_key_keyring_first(monkeypatch):
    """Verify that get_api_key checks keyring before env variables."""
    # Create mock keyring and set module attribute
    mock_keyring = MagicMock()
    mock_keyring.get_password.return_value = "abc"
    
    monkeypatch.setattr(secrets, "_KEYRING_AVAILABLE", True)
    monkeypatch.setattr(secrets, "keyring", mock_keyring, raising=False)
    
    monkeypatch.setenv("OPENAI_API_KEY", "env")
    val = secrets.get_api_key("openai")
    assert val == "abc"
    mock_keyring.get_password.assert_called_once_with("kairo", "openai")

def test_set_api_key_keyring(monkeypatch):
    """Verify that set_api_key attempts to store the key in keyring."""
    mock_keyring = MagicMock()
    
    monkeypatch.setattr(secrets, "_KEYRING_AVAILABLE", True)
    monkeypatch.setattr(secrets, "keyring", mock_keyring, raising=False)
    
    secrets.set_api_key("openai", "secret-val")
    mock_keyring.set_password.assert_called_once_with("kairo", "openai", "secret-val")

def test_set_api_key_no_keyring_no_crash(monkeypatch):
    """Verify set_api_key does not raise if keyring is unavailable."""
    monkeypatch.setattr(secrets, "_KEYRING_AVAILABLE", False)
    # Should execute silently without crashing
    secrets.set_api_key("openai", "secret-val")
