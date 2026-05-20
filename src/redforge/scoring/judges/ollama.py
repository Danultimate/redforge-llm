"""OllamaJudge — local judge for privacy-sensitive scans. DESIGN.md §6.4.

Hits the Ollama native HTTP API at `/api/generate`. Useful when running
RedForge against models whose system prompts are IP, contain customer data,
or are subject to compliance constraints — no scan content leaves your
network. Requires Ollama running locally (https://ollama.com).
"""

from __future__ import annotations

from typing import Any, ClassVar

from redforge.scoring.base import Judge
from redforge.scoring.judges._common import parse_verdict_json
from redforge.types import AttackPrompt, Severity, TargetResponse, Verdict


class OllamaJudge(Judge):
    model: ClassVar[str] = "llama3.1:70b"

    def __init__(
        self,
        model: str = "llama3.1:70b",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout_s: float = 120.0,
        client: Any | None = None,
    ) -> None:
        """If `client` is provided (testing), it must expose async `get(url)`
        and `post(url, json=...)` methods returning an httpx.Response-like
        object. Otherwise, a fresh httpx.AsyncClient is created per call.
        """
        self._model = model
        self._host = host.rstrip("/")
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._injected_client = client

    def _new_client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "OllamaJudge requires the `httpx` package. "
                "Install with: pip install redforge-llm[ollama]"
            ) from e
        return httpx.AsyncClient(timeout=self._timeout_s)

    async def evaluate(
        self,
        prompt: AttackPrompt,
        response: TargetResponse,
        rubric: str,
    ) -> Verdict:
        body = {
            "model": self._model,
            "prompt": rubric,
            "stream": False,
            "options": {"temperature": self._temperature},
        }
        async with _maybe_async_context(self._new_client()) as client:
            http_response = await client.post(f"{self._host}/api/generate", json=body)
            http_response.raise_for_status()
            data = http_response.json()
        text = str(data.get("response", "")).strip()
        if not text:
            from redforge.scoring.judges._common import JudgeParseError

            raise JudgeParseError("Ollama returned empty response.")
        parsed = parse_verdict_json(text)
        return Verdict(
            severity=Severity(parsed["severity"]),
            confidence=float(parsed["confidence"]),
            reason=str(parsed["reason"]),
            scored_by="judge",
            judge_model=self._model,
        )

    async def healthcheck(self) -> bool:
        """GET `/api/tags` to verify the Ollama host is reachable."""
        try:
            async with _maybe_async_context(self._new_client()) as client:
                http_response = await client.get(f"{self._host}/api/tags")
                http_response.raise_for_status()
        except Exception:
            return False
        return True

    def preflight(self) -> None:
        """Sync no-op. Use `await judge.healthcheck()` explicitly before scans
        if you want pre-flight validation; the Scanner does not auto-call it."""


class _maybe_async_context:
    """Wrap an httpx.AsyncClient (or fake) so it works as an async context
    manager whether or not it natively supports __aenter__/__aexit__."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._needs_close = False

    async def __aenter__(self) -> Any:
        # httpx.AsyncClient supports `async with` natively.
        if hasattr(self._client, "__aenter__"):
            self._entered = await self._client.__aenter__()
            return self._entered
        # Fake client passed for tests — use as-is, no close needed.
        return self._client

    async def __aexit__(self, *exc: Any) -> None:
        if hasattr(self._client, "__aexit__"):
            await self._client.__aexit__(*exc)
