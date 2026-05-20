"""Convenience helpers wrapping common LLM SDKs into TargetCallables.

These are DX sugar (DESIGN.md §6.1) — not core abstractions. Failing to maintain
one of them does not break the library; users can always write their own callable.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from redforge.types import TargetResponse


def from_anthropic(
    client: Any,
    model: str,
    system: str | None = None,
    max_tokens: int = 1024,
    extra_params: dict[str, Any] | None = None,
) -> Callable[[str], Awaitable[TargetResponse]]:
    """Wrap an AsyncAnthropic client into a TargetCallable.

    The returned callable awaits `client.messages.create(...)` and returns a
    TargetResponse with:

      - text concatenated from every text block in the response
      - metadata: tokens_in, tokens_out, latency_ms, stop_reason, model

    Errors are not caught here — the Executor's retry layer handles transient
    failures and the Scanner records them as AttackResult.error.
    """
    extras = dict(extra_params or {})

    async def call(prompt: str) -> TargetResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            **extras,
        }
        if system is not None:
            kwargs["system"] = system

        started = time.perf_counter()
        message = await client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - started) * 1000

        text = _join_text_blocks(message)
        metadata: dict[str, Any] = {
            "latency_ms": latency_ms,
            "model": getattr(message, "model", model),
            "stop_reason": getattr(message, "stop_reason", None),
        }
        usage = getattr(message, "usage", None)
        if usage is not None:
            tokens_in = getattr(usage, "input_tokens", None)
            tokens_out = getattr(usage, "output_tokens", None)
            if tokens_in is not None:
                metadata["tokens_in"] = int(tokens_in)
            if tokens_out is not None:
                metadata["tokens_out"] = int(tokens_out)

        return TargetResponse(text=text, metadata=metadata)

    return call


def from_openai(
    client: Any,
    model: str,
    system: str | None = None,
    max_tokens: int = 1024,
) -> Callable[[str], Awaitable[TargetResponse]]:
    """Wrap an AsyncOpenAI client into a Target."""
    raise NotImplementedError


def from_ollama(
    host: str,
    model: str,
    system: str | None = None,
) -> Callable[[str], Awaitable[TargetResponse]]:
    """Wrap an Ollama-compatible HTTP endpoint into a Target."""
    raise NotImplementedError


def _join_text_blocks(message: Any) -> str:
    """Concatenate text from every text-bearing content block in an Anthropic
    Message. Returns empty string if there are no text blocks (e.g. tool-only
    responses)."""
    content = getattr(message, "content", None) or []
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)
