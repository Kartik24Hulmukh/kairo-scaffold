"""F1 — Tier inference client for Ollama/llama.cpp OpenAI-compatible gateway.

Routes inference requests to a local Ollama or llama.cpp server exposing the
OpenAI-compatible chat completions API at :4000 (configurable).

Two model slots:
    Tier-1 (worker):  small, fast SLM — default: "llama3.2:3b"
    Tier-2 (reasoner): larger model  — default: "llama3.1:8b"

Endpoint: POST http://localhost:4000/v1/chat/completions
          (OpenAI-compatible; same format as Ollama REST API)

Offline / CI behaviour:
    When KAIRO_TIER_OFFLINE=1 or the server is unreachable, TierClient
    returns a deterministic stub response "[offline: model=...]" instead of
    raising an exception. This keeps tests hermetic without requiring a live
    model server.

Security / Verifier Independence:
    This module MUST NOT be imported by kernel/sidecar/models/vgva.py.
    The grounding verifier (VGVA) is model-independent and must remain so.
    The AST import scan in test_tiered_router.py::test_verifier_independence
    enforces this constraint at test time.

Usage::

    client = TierClient()
    response = client.complete_tier1("What is the invoice total?")
    # → "The invoice total is $1,234.56." (or stub in offline mode)

    response = client.complete_tier2("Compare warranty terms across contracts.")
    # → "Tier-2 response..." (or stub in offline mode)

    response = client.complete("llama3.2:3b", messages=[{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------

GATEWAY_BASE_URL: str = os.environ.get("KAIRO_GATEWAY_URL", "http://localhost:4000")
TIER1_MODEL: str = os.environ.get("KAIRO_TIER1_MODEL", "llama3.2:3b")
TIER2_MODEL: str = os.environ.get("KAIRO_TIER2_MODEL", "llama3.1:8b")
REQUEST_TIMEOUT_S: float = float(os.environ.get("KAIRO_GATEWAY_TIMEOUT_S", "30.0"))
OFFLINE_MODE: bool = os.environ.get("KAIRO_TIER_OFFLINE", "").lower() in (
    "1", "true", "yes",
)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _user_message(content: str) -> dict:
    return {"role": "user", "content": content}


def _system_message(content: str) -> dict:
    return {"role": "system", "content": content}


# ---------------------------------------------------------------------------
# TierClient
# ---------------------------------------------------------------------------


class TierClient:
    """HTTP client for the Ollama/llama.cpp OpenAI-compatible gateway.

    Sends chat completion requests to the configured gateway URL.
    Falls back to offline stub when the server is unreachable or
    KAIRO_TIER_OFFLINE=1 is set.

    Usage::

        client = TierClient()
        text = client.complete_tier1("Extract the invoice total from: ...")
        text = client.complete_tier2("Synthesize findings across both contracts.")

    The grounding verifier (vgva.py) must NOT import this module.
    """

    def __init__(
        self,
        base_url: str = GATEWAY_BASE_URL,
        tier1_model: str = TIER1_MODEL,
        tier2_model: str = TIER2_MODEL,
        timeout_s: float = REQUEST_TIMEOUT_S,
        offline: bool = OFFLINE_MODE,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._tier1_model = tier1_model
        self._tier2_model = tier2_model
        self._timeout_s = timeout_s
        self._offline = offline

    @property
    def tier1_model(self) -> str:
        return self._tier1_model

    @property
    def tier2_model(self) -> str:
        return self._tier2_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete_tier1(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
    ) -> str:
        """Send a prompt to the Tier-1 (worker/SLM) model.

        Args:
            prompt: The user prompt.
            system: Optional system message.
            temperature: Sampling temperature (0.0 for deterministic).

        Returns:
            The model's response text, or a stub string in offline mode.
        """
        messages = self._build_messages(prompt, system)
        return self.complete(self._tier1_model, messages, temperature=temperature)

    def complete_tier2(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
    ) -> str:
        """Send a prompt to the Tier-2 (reasoner/LLM) model.

        Args:
            prompt: The user prompt.
            system: Optional system message.
            temperature: Sampling temperature (0.3 for reasoning).

        Returns:
            The model's response text, or a stub string in offline mode.
        """
        messages = self._build_messages(prompt, system)
        return self.complete(self._tier2_model, messages, temperature=temperature)

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat completion request to the gateway.

        Args:
            model: Model name/tag (e.g., "llama3.2:3b").
            messages: List of {role, content} dicts.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.

        Returns:
            The assistant message content string.
        """
        if self._offline:
            return self._offline_stub(model, messages)

        try:
            return self._http_complete(model, messages, temperature, max_tokens)
        except Exception as exc:
            # Graceful fallback — never crash the pipeline
            return self._offline_stub(model, messages, error=str(exc))

    # ------------------------------------------------------------------
    # Internal HTTP request (stdlib only — no requests/httpx dependency)
    # ------------------------------------------------------------------

    def _http_complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """POST to the OpenAI-compatible chat completions endpoint.

        Uses urllib.request (stdlib) — no external HTTP dependency.
        """
        import urllib.error
        import urllib.request

        url = f"{self._base_url}/v1/chat/completions"
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            body = resp.read().decode("utf-8")

        data: dict[str, Any] = json.loads(body)
        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"No choices in response: {body[:200]}")
        return choices[0]["message"]["content"]

    # ------------------------------------------------------------------
    # Offline stub
    # ------------------------------------------------------------------

    @staticmethod
    def _offline_stub(
        model: str,
        messages: list[dict[str, str]],
        error: Optional[str] = None,
    ) -> str:
        """Return a deterministic offline stub response."""
        last_content = messages[-1]["content"] if messages else ""
        tag = f"offline: model={model}"
        if error:
            tag += f", error={error[:60]}"
        return f"[{tag}] echo: {last_content[:80]}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        prompt: str,
        system: Optional[str],
    ) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append(_system_message(system))
        msgs.append(_user_message(prompt))
        return msgs

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self, timeout_s: float = 2.0) -> bool:
        """Return True if the gateway is reachable within timeout_s seconds."""
        import urllib.error
        import urllib.request

        try:
            url = f"{self._base_url}/v1/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_s):
                return True
        except Exception:
            return False
