"""AnthropicJudge — default cloud judge. DESIGN.md §6.4."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar

from redforge.scoring.base import Judge
from redforge.scoring.judges._common import (
    JudgeParseError,
    parse_verdict_json,
)
from redforge.types import AttackPrompt, Severity, TargetResponse, Verdict

if TYPE_CHECKING:  # pragma: no cover
    from anthropic import AsyncAnthropic

# Re-export for any direct importers of this module.
__all__ = ["AnthropicJudge", "JudgeParseError"]


class AnthropicJudge(Judge):
    """LLM-as-judge using Anthropic's Claude (default: Haiku 4.5).

    The client is injected (for testing) or lazy-constructed from the
    `api_key` constructor arg / `ANTHROPIC_API_KEY` environment variable.
    """

    model: ClassVar[str] = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        temperature: float = 0.0,
        max_tokens: int = 256,
        api_key: str | None = None,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._client: AsyncAnthropic | None = client

    def _get_client(self) -> AsyncAnthropic:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "AnthropicJudge requires the `anthropic` package. "
                    "Install with: pip install 'redforge-llm[anthropic]'"
                ) from e
            key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError(
                    "No Anthropic API key. Set ANTHROPIC_API_KEY in your environment, "
                    "pass api_key=..., or pass judge=None to Scanner for heuristic-only "
                    "scoring (higher false-positive rate)."
                )
            self._client = AsyncAnthropic(api_key=key)
        return self._client

    async def evaluate(
        self,
        prompt: AttackPrompt,
        response: TargetResponse,
        rubric: str,
    ) -> Verdict:
        client = self._get_client()
        message = await client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": rubric}],
        )
        text = _extract_text(message)
        parsed = parse_verdict_json(text)
        return Verdict(
            severity=Severity(parsed["severity"]),
            confidence=float(parsed["confidence"]),
            reason=str(parsed["reason"]),
            scored_by="judge",
            judge_model=self._model,
        )

    async def healthcheck(self) -> bool:
        """Minimal API call to verify connectivity and auth. ~10 tokens."""
        client = self._get_client()
        try:
            await client.messages.create(
                model=self._model,
                max_tokens=5,
                messages=[{"role": "user", "content": "ping"}],
            )
        except Exception:
            return False
        return True

    def preflight(self) -> None:
        """Validate credentials are available without making an API call."""
        if self._client is not None:
            return  # Injected client; trust it.
        key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "No Anthropic API key found. RedForge's default judge requires "
                "ANTHROPIC_API_KEY in your environment.\n\n"
                "Fix one of:\n"
                "  1. Export ANTHROPIC_API_KEY=... and retry.\n"
                "  2. Pass api_key=... to AnthropicJudge().\n"
                "  3. Use heuristic-only scoring (faster, free, higher false-positive\n"
                "     rate): Scanner(target=..., scorer=DefaultScorer(judge=None))"
            )


def _extract_text(message: Any) -> str:
    """Pull the text payload out of an Anthropic Message object."""
    content = getattr(message, "content", None)
    if not content:
        raise JudgeParseError("Judge returned a message with no content.")
    # content is a list of blocks; first text block wins.
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            return text
    raise JudgeParseError("Judge returned no text blocks.")
