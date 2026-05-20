"""Unit tests for OllamaJudge — uses an injected fake httpx-like client to
avoid real network calls.
"""

from __future__ import annotations

import pytest

from redforge.scoring.judges._common import JudgeParseError
from redforge.scoring.judges.ollama import OllamaJudge
from redforge.types import AttackPrompt, Severity, TargetResponse


class _FakeResponse:
    """Mimics the slice of httpx.Response we use."""

    def __init__(
        self, *, status_code: int = 200, payload: dict | None = None
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Records calls and returns scripted responses."""

    def __init__(
        self,
        *,
        post_response: _FakeResponse | None = None,
        get_response: _FakeResponse | None = None,
    ) -> None:
        self.post_response = post_response or _FakeResponse(
            payload={"response": '{"severity":"passed","confidence":0.9,"reason":"r"}'}
        )
        self.get_response = get_response or _FakeResponse(payload={"models": []})
        self.last_post_url: str | None = None
        self.last_post_json: dict | None = None
        self.last_get_url: str | None = None

    async def post(self, url: str, json: dict) -> _FakeResponse:
        self.last_post_url = url
        self.last_post_json = json
        return self.post_response

    async def get(self, url: str) -> _FakeResponse:
        self.last_get_url = url
        return self.get_response


def _prompt() -> AttackPrompt:
    return AttackPrompt(
        id="x", module="PromptInjection", variant="DirectOverride", prompt="..."
    )


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_returns_judge_verdict(self) -> None:
        client = _FakeClient(
            post_response=_FakeResponse(
                payload={
                    "response": (
                        '{"severity":"critical","confidence":0.91,"reason":"leak"}'
                    )
                }
            )
        )
        judge = OllamaJudge(client=client)
        verdict = await judge.evaluate(
            _prompt(), TargetResponse(text="resp"), rubric="rubric"
        )
        assert verdict.severity == Severity.CRITICAL
        assert verdict.confidence == 0.91
        assert verdict.scored_by == "judge"
        assert verdict.judge_model == "llama3.1:70b"

    @pytest.mark.asyncio
    async def test_posts_to_generate_endpoint(self) -> None:
        client = _FakeClient()
        judge = OllamaJudge(client=client, host="http://localhost:11434")
        await judge.evaluate(_prompt(), TargetResponse(text="x"), rubric="rubric body")
        assert client.last_post_url == "http://localhost:11434/api/generate"
        assert client.last_post_json is not None
        assert client.last_post_json["model"] == "llama3.1:70b"
        assert client.last_post_json["stream"] is False
        assert client.last_post_json["options"]["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_trailing_slash_on_host_normalised(self) -> None:
        client = _FakeClient()
        judge = OllamaJudge(client=client, host="http://localhost:11434/")
        await judge.evaluate(_prompt(), TargetResponse(text="x"), rubric="r")
        # No double slash.
        assert client.last_post_url == "http://localhost:11434/api/generate"

    @pytest.mark.asyncio
    async def test_custom_model_propagated(self) -> None:
        client = _FakeClient()
        judge = OllamaJudge(client=client, model="qwen2.5:7b")
        await judge.evaluate(_prompt(), TargetResponse(text="x"), rubric="r")
        assert client.last_post_json is not None
        assert client.last_post_json["model"] == "qwen2.5:7b"

    @pytest.mark.asyncio
    async def test_empty_response_raises(self) -> None:
        client = _FakeClient(
            post_response=_FakeResponse(payload={"response": ""})
        )
        judge = OllamaJudge(client=client)
        with pytest.raises(JudgeParseError):
            await judge.evaluate(_prompt(), TargetResponse(text="x"), rubric="r")

    @pytest.mark.asyncio
    async def test_unparseable_response_raises(self) -> None:
        client = _FakeClient(
            post_response=_FakeResponse(
                payload={"response": "not JSON, just prose"}
            )
        )
        judge = OllamaJudge(client=client)
        with pytest.raises(JudgeParseError):
            await judge.evaluate(_prompt(), TargetResponse(text="x"), rubric="r")

    @pytest.mark.asyncio
    async def test_http_error_propagates(self) -> None:
        client = _FakeClient(
            post_response=_FakeResponse(status_code=500, payload={})
        )
        judge = OllamaJudge(client=client)
        with pytest.raises(RuntimeError):
            await judge.evaluate(_prompt(), TargetResponse(text="x"), rubric="r")


class TestHealthcheck:
    @pytest.mark.asyncio
    async def test_returns_true_when_tags_endpoint_responds(self) -> None:
        client = _FakeClient(get_response=_FakeResponse(payload={"models": ["llama3.1:70b"]}))
        judge = OllamaJudge(client=client)
        assert await judge.healthcheck() is True
        assert client.last_get_url == "http://localhost:11434/api/tags"

    @pytest.mark.asyncio
    async def test_returns_false_on_failure(self) -> None:
        client = _FakeClient(get_response=_FakeResponse(status_code=503))
        judge = OllamaJudge(client=client)
        assert await judge.healthcheck() is False


class TestPreflight:
    def test_is_a_noop(self) -> None:
        """OllamaJudge.preflight is documented as sync no-op; the real check
        is async healthcheck() which must be called explicitly."""
        OllamaJudge(client=_FakeClient()).preflight()
        OllamaJudge(host="http://nonexistent:99999").preflight()
