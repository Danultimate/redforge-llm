"""Unit tests for AnthropicJudge — covers the JSON parser and the preflight
credential check. Real API calls are exercised by env-gated integration tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from redforge.scoring.judges._common import (
    JudgeParseError,
)
from redforge.scoring.judges._common import (
    parse_verdict_json as _parse_verdict_json,
)
from redforge.scoring.judges.anthropic import (
    AnthropicJudge,
    _extract_text,
)
from redforge.types import AttackPrompt, Severity, TargetResponse


class TestParseVerdictJson:
    def test_plain_json(self) -> None:
        out = _parse_verdict_json(
            '{"severity":"critical","confidence":0.92,"reason":"leaked"}'
        )
        assert out["severity"] == "critical"
        assert out["confidence"] == 0.92
        assert out["reason"] == "leaked"

    def test_strips_markdown_fence(self) -> None:
        text = '```json\n{"severity":"passed","confidence":0.9,"reason":"refused"}\n```'
        out = _parse_verdict_json(text)
        assert out["severity"] == "passed"

    def test_extracts_object_from_noisy_prose(self) -> None:
        text = (
            "Sure! Here is my verdict: "
            '{"severity":"high","confidence":0.7,"reason":"partial leak"} '
            "Hope that helps."
        )
        out = _parse_verdict_json(text)
        assert out["severity"] == "high"

    def test_uppercase_severity_normalised(self) -> None:
        out = _parse_verdict_json('{"severity":"CRITICAL","confidence":0.9,"reason":"x"}')
        assert out["severity"] == "critical"

    def test_unknown_severity_raises(self) -> None:
        with pytest.raises(JudgeParseError):
            _parse_verdict_json('{"severity":"explosive","confidence":0.9,"reason":"x"}')

    def test_missing_key_raises(self) -> None:
        with pytest.raises(JudgeParseError):
            _parse_verdict_json('{"severity":"high","confidence":0.9}')

    def test_garbage_raises(self) -> None:
        with pytest.raises(JudgeParseError):
            _parse_verdict_json("the model produced no JSON whatsoever")


class TestExtractText:
    def test_returns_first_text_block(self) -> None:
        message = SimpleNamespace(content=[SimpleNamespace(text="hello world")])
        assert _extract_text(message) == "hello world"

    def test_skips_empty_blocks(self) -> None:
        message = SimpleNamespace(
            content=[
                SimpleNamespace(text=""),
                SimpleNamespace(text="real text"),
            ]
        )
        assert _extract_text(message) == "real text"

    def test_empty_content_raises(self) -> None:
        message = SimpleNamespace(content=[])
        with pytest.raises(JudgeParseError):
            _extract_text(message)


class TestPreflight:
    def test_raises_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        judge = AnthropicJudge()
        with pytest.raises(RuntimeError) as exc_info:
            judge.preflight()
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)
        # Error message must be actionable: tell the user how to fix.
        assert "judge=None" in str(exc_info.value)

    def test_accepts_explicit_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        judge = AnthropicJudge(api_key="sk-test-fake")
        judge.preflight()  # Must not raise.

    def test_accepts_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
        judge = AnthropicJudge()
        judge.preflight()

    def test_injected_client_skips_check(self) -> None:
        """When tests inject a client, preflight should not require credentials."""
        judge = AnthropicJudge(client=_FakeClient())
        judge.preflight()


_DEFAULT_FAKE_RESPONSE = '{"severity":"passed","confidence":0.9,"reason":"r"}'


class _FakeClient:
    """Minimal stub mimicking AsyncAnthropic for evaluate() tests."""

    def __init__(self, response_text: str = _DEFAULT_FAKE_RESPONSE) -> None:
        self._response_text = response_text
        self.messages = _FakeMessages(response_text)


class _FakeMessages:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        return SimpleNamespace(content=[SimpleNamespace(text=self._response_text)])


class TestEvaluateWithInjectedClient:
    @pytest.mark.asyncio
    async def test_evaluate_returns_judge_verdict(self) -> None:
        client = _FakeClient('{"severity":"critical","confidence":0.95,"reason":"full leak"}')
        judge = AnthropicJudge(client=client)
        prompt = AttackPrompt(
            id="x", module="PromptInjection", variant="DirectOverride", prompt="..."
        )
        response = TargetResponse(text="You are a helpful bot...")
        verdict = await judge.evaluate(prompt, response, rubric="...rendered template...")
        assert verdict.severity == Severity.CRITICAL
        assert verdict.confidence == 0.95
        assert verdict.scored_by == "judge"
        assert verdict.judge_model == judge._model

    @pytest.mark.asyncio
    async def test_evaluate_calls_temperature_zero(self) -> None:
        client = _FakeClient()
        judge = AnthropicJudge(client=client, temperature=0.0)
        prompt = AttackPrompt(
            id="x", module="PromptInjection", variant="DirectOverride", prompt="..."
        )
        await judge.evaluate(prompt, TargetResponse(text="ok"), rubric="r")
        assert client.messages.last_kwargs is not None
        assert client.messages.last_kwargs["temperature"] == 0.0
