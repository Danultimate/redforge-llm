"""Unit tests for OpenAIJudge — covers the response extractor, the parser
path, and the preflight credential check. Uses an injected fake client; no
real API calls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from redforge.scoring.judges._common import JudgeParseError
from redforge.scoring.judges.openai import OpenAIJudge, _extract_text
from redforge.types import AttackPrompt, Severity, TargetResponse


def _completion(text: str) -> Any:
    """Build an object that mimics openai.ChatCompletion's structure."""
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


class _FakeChatCompletions:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        return _completion(self._response_text)


class _FakeChat:
    def __init__(self, response_text: str) -> None:
        self.completions = _FakeChatCompletions(response_text)


class _FakeClient:
    def __init__(
        self,
        response_text: str = '{"severity":"passed","confidence":0.9,"reason":"refused"}',
    ) -> None:
        self.chat = _FakeChat(response_text)


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_extracts_first_choice_message_content(self) -> None:
        completion = _completion("hello world")
        assert _extract_text(completion) == "hello world"

    def test_missing_choices_raises(self) -> None:
        with pytest.raises(JudgeParseError):
            _extract_text(SimpleNamespace(choices=[]))

    def test_missing_message_raises(self) -> None:
        completion = SimpleNamespace(choices=[SimpleNamespace(message=None)])
        with pytest.raises(JudgeParseError):
            _extract_text(completion)

    def test_empty_text_raises(self) -> None:
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
        )
        with pytest.raises(JudgeParseError):
            _extract_text(completion)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


class TestPreflight:
    def test_raises_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        judge = OpenAIJudge()
        with pytest.raises(RuntimeError) as exc_info:
            judge.preflight()
        msg = str(exc_info.value)
        assert "OPENAI_API_KEY" in msg
        # Error must point users at the available fallbacks.
        assert "judge=None" in msg or "AnthropicJudge" in msg

    def test_accepts_explicit_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        judge = OpenAIJudge(api_key="sk-test-fake")
        judge.preflight()

    def test_accepts_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
        OpenAIJudge().preflight()

    def test_injected_client_skips_check(self) -> None:
        judge = OpenAIJudge(client=_FakeClient())
        judge.preflight()  # must not raise


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluateWithInjectedClient:
    @pytest.mark.asyncio
    async def test_returns_judge_verdict(self) -> None:
        client = _FakeClient(
            '{"severity":"critical","confidence":0.95,"reason":"full leak"}'
        )
        judge = OpenAIJudge(client=client)
        prompt = AttackPrompt(
            id="x", module="PromptInjection", variant="DirectOverride", prompt="..."
        )
        response = TargetResponse(text="You are a helpful bot...")
        verdict = await judge.evaluate(prompt, response, rubric="rubric text")
        assert verdict.severity == Severity.CRITICAL
        assert verdict.confidence == 0.95
        assert verdict.scored_by == "judge"
        assert verdict.judge_model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_temperature_zero(self) -> None:
        client = _FakeClient()
        judge = OpenAIJudge(client=client, temperature=0.0)
        prompt = AttackPrompt(
            id="x", module="PromptInjection", variant="DirectOverride", prompt="..."
        )
        await judge.evaluate(prompt, TargetResponse(text="ok"), rubric="r")
        assert client.chat.completions.last_kwargs is not None
        assert client.chat.completions.last_kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_json_mode_enabled_by_default(self) -> None:
        client = _FakeClient()
        judge = OpenAIJudge(client=client)
        prompt = AttackPrompt(
            id="x", module="PromptInjection", variant="DirectOverride", prompt="..."
        )
        await judge.evaluate(prompt, TargetResponse(text="ok"), rubric="r")
        kwargs = client.chat.completions.last_kwargs
        assert kwargs is not None
        assert kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_json_mode_can_be_disabled(self) -> None:
        client = _FakeClient()
        judge = OpenAIJudge(client=client, use_json_mode=False)
        prompt = AttackPrompt(
            id="x", module="PromptInjection", variant="DirectOverride", prompt="..."
        )
        await judge.evaluate(prompt, TargetResponse(text="ok"), rubric="r")
        kwargs = client.chat.completions.last_kwargs
        assert kwargs is not None
        assert "response_format" not in kwargs

    @pytest.mark.asyncio
    async def test_unparseable_response_raises(self) -> None:
        client = _FakeClient("this is not JSON at all")
        judge = OpenAIJudge(client=client)
        prompt = AttackPrompt(
            id="x", module="PromptInjection", variant="DirectOverride", prompt="..."
        )
        with pytest.raises(JudgeParseError):
            await judge.evaluate(prompt, TargetResponse(text="ok"), rubric="r")
