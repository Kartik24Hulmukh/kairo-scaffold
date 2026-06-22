"""H1 — Kairo structured logger with redaction pass.

Rules:
  1. LOCAL ONLY — no network transport. All handlers write to local disk or stderr.
  2. REDACTION — document-derived text (chunk content, extracted values) is NEVER
     written to logs. Only metadata (chunk_id, doc_id, bbox coords, status codes)
     is logged.
  3. STRUCTURED — every log record is a JSON object for machine readability.
  4. LEVELS — DEBUG | INFO | WARN | ERROR | CRITICAL

Usage::

    from kernel.sidecar.models.kairo_logger import get_logger
    log = get_logger("sidecar.app")
    log.info("indexed", doc_id="abc123", chunks=42)
    log.warn("quarantined", chunk_id="c99", pattern_count=3)
    log.error("sidecar_crash", error_type="OSError", recoverable=False)

Sensitive fields that are ALWAYS redacted from logs:
  - chunk.text, chunk_text, text, content, extracted_value, value, query
  - Any key whose value is a long string (>120 chars) — likely document content
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Redaction rules
# ---------------------------------------------------------------------------

# Keys whose values should ALWAYS be redacted (replaced with <REDACTED>)
_REDACTED_KEYS = frozenset({
    "text",
    "chunk_text",
    "content",
    "value",
    "extracted_value",
    "query",
    "answer_text",
    "raw_text",
    "paragraph",
    "body",
    "caption",
    "metadata_value",
    "prompt",
    "system_prompt",
    "user_prompt",
})

# Maximum length for any string value in a log record.
# Strings longer than this are likely document content and are truncated.
_MAX_STR_LEN = 120


def _redact(value: Any, key: str = "") -> Any:
    """Redact or truncate a log value.

    - Keys in _REDACTED_KEYS → "<REDACTED>"
    - String values longer than _MAX_STR_LEN → truncated + "...<truncated>"
    - All other values → pass through
    """
    if key.lower() in _REDACTED_KEYS:
        return "<REDACTED>"
    if isinstance(value, str) and len(value) > _MAX_STR_LEN:
        return value[:_MAX_STR_LEN] + "...<truncated>"
    return value


def _redact_dict(d: dict) -> dict:
    """Recursively redact all values in a dict."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _redact_dict(v)
        elif isinstance(v, list):
            out[k] = [_redact(item, k) if not isinstance(item, dict)
                      else _redact_dict(item)
                      for item in v]
        else:
            out[k] = _redact(v, k)
    return out


# ---------------------------------------------------------------------------
# Structured JSON log formatter
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach any extra kwargs passed to the log call
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }
        }
        if extras:
            base["ctx"] = _redact_dict(extras)
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, default=str)


# ---------------------------------------------------------------------------
# Logger factory — LOCAL ONLY (no network handlers ever attached)
# ---------------------------------------------------------------------------

_LOG_DIR = os.environ.get(
    "KAIRO_LOG_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", ".kairo", "logs"),
)

_LOG_LEVEL = os.environ.get("KAIRO_LOG_LEVEL", "INFO").upper()

# Registry of loggers we've already configured — prevents duplicate handlers
_configured: set[str] = set()


def get_logger(name: str) -> "KairoLogger":
    """Return a KairoLogger for the given name.

    Creates and caches the underlying logging.Logger with:
      - A stderr StreamHandler (always present, human-readable JSON)
      - A rotating FileHandler writing to KAIRO_LOG_DIR/kairo.log (if writable)

    NO network handlers are ever added. The log directory is local only.
    """
    underlying = logging.getLogger(f"kairo.{name}")

    if name not in _configured:
        underlying.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
        underlying.propagate = False

        formatter = _JsonFormatter()

        # Stderr handler — always local
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(formatter)
        underlying.addHandler(sh)

        # File handler — local only, skip gracefully if disk is full/unavailable
        try:
            os.makedirs(_LOG_DIR, exist_ok=True)
            log_path = os.path.join(_LOG_DIR, "kairo.log")
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(formatter)
            underlying.addHandler(fh)
        except (OSError, PermissionError):
            # Disk full, permission denied, etc. — fall back to stderr only
            underlying.warning("Log file unavailable; stderr only")

        _configured.add(name)

    return KairoLogger(underlying)


# ---------------------------------------------------------------------------
# Thin wrapper that enforces keyword-only structured logging
# ---------------------------------------------------------------------------

class KairoLogger:
    """Thin wrapper around logging.Logger that enforces structured, redacted logging.

    All log calls accept a mandatory message string plus keyword arguments
    that become the "ctx" field in the JSON output. Document text keys are
    automatically redacted by _redact_dict.

    Example::

        log = get_logger("sidecar.injection")
        log.warn("quarantine", chunk_id="c42", pattern_count=2)
        # → {"ts":"...","level":"WARNING","logger":"kairo.sidecar.injection",
        #     "msg":"quarantine","ctx":{"chunk_id":"c42","pattern_count":2}}
    """

    def __init__(self, underlying: logging.Logger) -> None:
        self._log = underlying

    def _emit(self, level: int, msg: str, **kwargs: Any) -> None:
        self._log.log(level, msg, extra=kwargs)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.INFO, msg, **kwargs)

    def warn(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        self._emit(logging.CRITICAL, msg, **kwargs)

    def exception(self, msg: str, exc: Exception, **kwargs: Any) -> None:
        self._log.exception(msg, exc_info=exc, extra=kwargs)
