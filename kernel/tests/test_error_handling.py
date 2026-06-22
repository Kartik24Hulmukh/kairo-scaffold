"""G2 tests — Runtime error handling: exception hierarchy and format_user_error().

Covers:
  - KairoError is the base class
  - GroundingError, IndexError, SidecarUnavailable each raise correctly
  - format_user_error returns correct dict shape for each subclass (recoverable=True)
  - format_user_error returns recoverable=False for unknown exception types
"""
import sys
import pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from kernel.sidecar.models.error_handling import (
    KairoError,
    GroundingError,
    IndexError as KairoIndexError,
    SidecarUnavailable,
    format_user_error,
)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

def test_kairo_error_is_base_exception():
    exc = KairoError("base error")
    assert isinstance(exc, Exception)


def test_grounding_error_inherits_kairo_error():
    exc = GroundingError("no anchor found")
    assert isinstance(exc, KairoError)
    assert isinstance(exc, Exception)


def test_index_error_inherits_kairo_error():
    exc = KairoIndexError("document indexing failed")
    assert isinstance(exc, KairoError)


def test_sidecar_unavailable_inherits_kairo_error():
    exc = SidecarUnavailable("sidecar port 8765 not responding")
    assert isinstance(exc, KairoError)


# ---------------------------------------------------------------------------
# Each subclass raises correctly via raise statement
# ---------------------------------------------------------------------------

def test_grounding_error_can_be_raised_and_caught():
    with pytest.raises(GroundingError, match="no anchor"):
        raise GroundingError("no anchor for value 'INV-9001'")


def test_index_error_can_be_raised_and_caught():
    with pytest.raises(KairoIndexError, match="indexing failed"):
        raise KairoIndexError("indexing failed: file not found")


def test_sidecar_unavailable_can_be_raised_and_caught():
    with pytest.raises(SidecarUnavailable, match="port 8765"):
        raise SidecarUnavailable("sidecar port 8765 not responding")


def test_kairo_error_subclass_caught_by_kairo_error_handler():
    """A caller catching KairoError handles all subclasses."""
    with pytest.raises(KairoError):
        raise GroundingError("caught by base handler")


# ---------------------------------------------------------------------------
# format_user_error — dict shape
# ---------------------------------------------------------------------------

def test_format_grounding_error_recoverable():
    exc = GroundingError("extraction blocked")
    result = format_user_error(exc)
    assert result["error_type"] == "GroundingError"
    assert result["message"] == "extraction blocked"
    assert result["recoverable"] is True


def test_format_index_error_recoverable():
    exc = KairoIndexError("document not indexed")
    result = format_user_error(exc)
    assert result["error_type"] == "IndexError"
    assert result["recoverable"] is True


def test_format_sidecar_unavailable_recoverable():
    exc = SidecarUnavailable("offline")
    result = format_user_error(exc)
    assert result["error_type"] == "SidecarUnavailable"
    assert result["recoverable"] is True


def test_format_kairo_error_base_recoverable():
    exc = KairoError("generic kairo error")
    result = format_user_error(exc)
    assert result["recoverable"] is True


def test_format_unknown_exception_not_recoverable():
    exc = RuntimeError("unexpected crash")
    result = format_user_error(exc)
    assert result["error_type"] == "RuntimeError"
    assert result["message"] == "unexpected crash"
    assert result["recoverable"] is False


def test_format_value_error_not_recoverable():
    exc = ValueError("bad input")
    result = format_user_error(exc)
    assert result["recoverable"] is False


def test_format_user_error_returns_dict_with_required_keys():
    exc = KairoError("test")
    result = format_user_error(exc)
    assert set(result.keys()) == {"error_type", "message", "recoverable"}
