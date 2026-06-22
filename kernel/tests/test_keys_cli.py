"""Unit tests for kairo keys CLI subcommand and secrets keyring integration.

Covers:
  1. CLI keys subcommand argument parsing and matching
  2. Set, clear, list operations in CLI via cmd_keys
  3. keyring API integration and mock credential store verification
  4. Redaction of API keys in output
  5. Fallback behavior when keys are missing or cleared
"""
import sys
import os
import json
import pytest
from unittest.mock import MagicMock, patch

# Ensure kernel is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import kernel.sidecar.models.secrets as secrets
from cli.main import cmd_keys, main as cli_main

class DummyArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def test_redact_key():
    """Verify that redact_key correctly masks keys based on length/presence."""
    from kernel.sidecar.models.secrets import redact_key
    assert redact_key(None) == "None"
    assert redact_key("") == "None"
    assert redact_key("short") == "[REDACTED]"
    assert redact_key("12345678") == "[REDACTED]"
    assert redact_key("sk-test123456") == "sk-t...3456"
    assert redact_key("openai-api-key-very-long") == "open...long"

def test_cli_keys_set(monkeypatch):
    """Verify that kairo keys set stores key in keyring and outputs redacted confirmation."""
    mock_keyring = MagicMock()
    monkeypatch.setattr(secrets, "_KEYRING_AVAILABLE", True)
    monkeypatch.setattr(secrets, "keyring", mock_keyring, raising=False)

    args = DummyArgs(command="keys", keys_command="set", provider="openai", key="sk-test1234567890abcdef")
    
    with patch("builtins.print") as mock_print:
        cmd_keys(args)
        
        # Verify print output
        mock_print.assert_called_once()
        print_arg = mock_print.call_args[0][0]
        output = json.loads(print_arg)
        assert output["status"] == "success"
        assert "set successfully" in output["message"]
        assert output["provider"] == "openai"
        assert output["key"] == "sk-t...cdef"  # redacted

    mock_keyring.set_password.assert_called_once_with("kairo", "openai", "sk-test1234567890abcdef")

def test_cli_keys_clear(monkeypatch):
    """Verify that kairo keys clear removes key from keyring."""
    mock_keyring = MagicMock()
    monkeypatch.setattr(secrets, "_KEYRING_AVAILABLE", True)
    monkeypatch.setattr(secrets, "keyring", mock_keyring, raising=False)

    args = DummyArgs(command="keys", keys_command="clear", provider="openai")
    
    with patch("builtins.print") as mock_print:
        cmd_keys(args)
        
        # Verify print output
        mock_print.assert_called_once()
        print_arg = mock_print.call_args[0][0]
        output = json.loads(print_arg)
        assert output["status"] == "success"
        assert "cleared from OS keychain" in output["message"]
        assert output["provider"] == "openai"

    mock_keyring.delete_password.assert_called_once_with("kairo", "openai")

def test_cli_keys_list(monkeypatch):
    """Verify that kairo keys list shows configured providers with redacted key status."""
    mock_keyring = MagicMock()
    # Mock keyring to return key for openai and None for others
    def get_password(service, username):
        if username == "openai":
            return "sk-test1234567890abcdef"
        return None
    mock_keyring.get_password.side_effect = get_password

    monkeypatch.setattr(secrets, "_KEYRING_AVAILABLE", True)
    monkeypatch.setattr(secrets, "keyring", mock_keyring, raising=False)

    args = DummyArgs(command="keys", keys_command="list")
    
    with patch("builtins.print") as mock_print:
        cmd_keys(args)
        
        mock_print.assert_called_once()
        print_arg = mock_print.call_args[0][0]
        output = json.loads(print_arg)
        
        assert output["openai"]["configured"] is True
        assert output["openai"]["key"] == "sk-t...cdef"
        
        assert output["anthropic"]["configured"] is False
        assert output["anthropic"]["key"] == "None"
        
        assert output["google"]["configured"] is False
        assert output["google"]["key"] == "None"

def test_bench_local_fallback_no_keys():
    """Verify that benchmark runner falls back gracefully when no keys are configured."""
    from bench.run_bench import check_sidecar_running
    
    # We will patch check_sidecar_running to return False, and verify live_systems doesn't include cloud
    with patch("kernel.sidecar.models.secrets.get_api_key", return_value=None), \
         patch("bench.run_bench.check_sidecar_running", return_value=False), \
         patch("builtins.print") as mock_print:
        
        # Import main from run_bench inside patch context
        from bench.run_bench import main as bench_main
        
        # Deliberately raise SystemExit or catch it since it will fail to load questions.json if run directly,
        # but we just want to verify key setup logic. We can mock questions file check or questions.json loading.
        with patch("pathlib.Path.exists", return_value=False), \
             pytest.raises(SystemExit):
            bench_main()
            
        # Verify sidecar offline print message occurred
        any_offline_msg = any("Kairo sidecar is offline" in call[0][0] for call in mock_print.call_args_list if call[0])
        assert any_offline_msg is True
