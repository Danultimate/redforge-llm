"""OpenAIJudge — DESIGN.md §6.4."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar

from redforge.scoring.base import Judge
from redforge.scoring.judges._common import JudgeParseError, parse_verdict_json
from redforge.types import AttackPrompt, Severity, TargetResponse, Verdict

if TYPE_CHECKING:  # pragma: no cover
    from openai import AsyncOpenAI


class OpenAIJudge(Judge):
    """LLM-as-judge using OpenAI's Chat Completions API (default: gpt-4o-mini).

    The client is injected (for testing) or lazy-constructed from the
    `api_key` constructor arg / `OPENAI_API_KEY` environment variable.

    By default, `response_format={"type": "json_object"}` is requested — this
    materially improves the chance that the model returns parseable JSON.
    The rubric must contain the word "json" for this to work; the shipped
    rubrics already satisfy that requirement.
    """

    model: ClassVar[str] = "gpt-4o-mini"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 256,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        use_json_mode: bool = True,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._client: AsyncOpenAI | None = client
        self._use_json_mode = use_json_mode

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "OpenAIJudge requires the `openai` package. "
                    "Install with: pip install redforge-llm[openai]"
                ) from e
            key = self._api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError(
                    "No OpenAI API key. Set OPENAI_API_KEY in your environment, "
                    "pass api_key=..., or pass judge=None to Scanner for "
                    "heuristic-only scoring (higher false-positive rate)."
                )
            self._client = AsyncOpenAI(api_key=key)
        return self._client

    async def evaluate(
        self,
        prompt: AttackPrompt,
        response: TargetResponse,
        rubric: str,
    ) -> Verdict:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": [{"role": "user", "content": rubric}],
        }
        if self._use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        completion = await client.chat.completions.create(**kwargs)
        text = _extract_text(completion)
        parsed = parse_verdict_json(text)
        return Verdict(
            severity=Severity(parsed["severity"]),
            confidence=float(parsed["confidence"]),
            reason=str(parsed["reason"]),
            scored_by="judge",
            judge_model=self._model,
        )

    async def healthcheck(self) -> bool:
        """Tiny chat completion to verify connectivity and auth. ~5 tokens."""
        client = self._get_client()
        try:
            await client.chat.completions.create(
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
            return
        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "No OpenAI API key found. OpenAIJudge requires OPENAI_API_KEY "
                "in your environment.\n\n"
                "Fix one of:\n"
                "  1. Export OPENAI_API_KEY=... and retry.\n"
                "  2. Pass api_key=... to OpenAIJudge().\n"
                "  3. Use a different judge (AnthropicJudge, OllamaJudge), or\n"
                "     heuristic-only scoring: DefaultScorer(judge=None)."
            )


def _extract_text(completion: Any) -> str:
    """Pull the text payload out of an OpenAI ChatCompletion object."""
    choices = getattr(completion, "choices", None)
    if not choices:
        raise JudgeParseError("Judge returned a completion with no choices.")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise JudgeParseError("Judge completion choice has no message.")
    text = getattr(message, "content", None)
    if not isinstance(text, str) or not text.strip():
        raise JudgeParseError("Judge returned empty or non-text content.")
    return text
