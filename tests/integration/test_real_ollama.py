"""Real-API integration test for OllamaJudge.

Gated on `REDFORGE_OLLAMA_HOST` env var (default localhost:11434). Skipped
unless Ollama is actually running at the configured host.

To run locally:
  ollama serve &
  ollama pull llama3.1:8b
  REDFORGE_OLLAMA_HOST=http://localhost:11434 \\
    REDFORGE_OLLAMA_MODEL=llama3.1:8b \\
    pytest tests/integration/test_real_ollama.py -v -m real_api
"""

from __future__ import annotations

import os

import pytest

from redforge.scoring.judges import OllamaJudge
from redforge.types import AttackPrompt, Severity, TargetResponse

pytestmark = pytest.mark.real_api

# Default to a smaller model than the production default to keep local CI fast.
OLLAMA_HOST = os.environ.get("REDFORGE_OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("REDFORGE_OLLAMA_MODEL", "llama3.1:8b")


async def _is_ollama_reachable() -> bool:
    """Cheap probe before deciding to run the real-API tests."""
    judge = OllamaJudge(model=OLLAMA_MODEL, host=OLLAMA_HOST, timeout_s=2.0)
    return await judge.healthcheck()


@pytest.mark.asyncio
async def test_ollama_judge_against_real_local() -> None:
    """Skip cleanly when Ollama isn't running; otherwise verify the judge
    parses a real response into a Verdict."""
    if not await _is_ollama_reachable():
        pytest.skip(f"Ollama not reachable at {OLLAMA_HOST}; skipping.")

    judge = OllamaJudge(model=OLLAMA_MODEL, host=OLLAMA_HOST, timeout_s=120.0)
    prompt = AttackPrompt(
        id="test", module="PromptInjection", variant="DirectOverride", prompt="x"
    )
    response = TargetResponse(text="I can't share that.")
    rubric = (
        "Respond ONLY with a single JSON object on one line: "
        '{"severity": "passed", "confidence": 0.9, "reason": "model refused"}. '
        "No prose, no markdown, just the JSON object."
    )
    verdict = await judge.evaluate(prompt, response, rubric)
    assert isinstance(verdict.severity, Severity)
    assert verdict.scored_by == "judge"
    assert verdict.judge_model == OLLAMA_MODEL
