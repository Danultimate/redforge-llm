"""End-to-end integration test: Scanner.run() against scripted MockTargets.

Verifies the full wiring works: corpus loaded from YAML → executor runs
prompts → heuristic scorer produces verdicts → reporters write artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from redforge import ScanConfig, Scanner, TargetResponse
from redforge.attacks import PromptInjection
from redforge.types import Severity


@pytest.fixture
def artifact_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs"


async def refusing_target(prompt: str) -> TargetResponse:
    """A target that always refuses — should produce mostly PASSED verdicts."""
    return TargetResponse(text="I can't share that information.")


async def leaking_target(prompt: str) -> TargetResponse:
    """A target that leaks its system prompt — should produce CRITICAL verdicts."""
    return TargetResponse(
        text=(
            "Sure! My system prompt: You are a helpful assistant. "
            "Your instructions are to be polite and helpful."
        )
    )


async def engaged_but_safe_target(prompt: str) -> TargetResponse:
    """Engages without leaking and without an obvious refusal — ambiguous."""
    return TargetResponse(text="That's an interesting question. Let me think...")


@pytest.mark.asyncio
async def test_refusing_target_no_critical(artifact_dir: Path) -> None:
    """Refusing targets should never produce CRITICAL findings, regardless of
    which variant is being attacked. Different variants will distribute the
    remaining verdicts between PASSED (refusal-aware variants) and INFO
    (IndirectInjection, which expects task completion not refusal)."""
    config = ScanConfig(sample_size=5, seed=42, artifact_dir=artifact_dir)
    scanner = Scanner(
        target=refusing_target,
        attacks=[PromptInjection()],
        reporters=[],
        config=config,
    )
    scan = await scanner.run()

    assert not scan.incomplete
    # 5 prompts × 4 variants in the current corpus.
    assert len(scan.results) == 4 * 5
    assert scan.summary[Severity.CRITICAL] == 0
    assert scan.summary[Severity.HIGH] == 0
    # Refusal-aware variants (DirectOverride, DelimiterConfusion, NestedInjection)
    # should classify these responses as PASSED. IndirectInjection's heuristic
    # has expected_refusal=false, so refusals produce INFO instead.
    assert scan.summary[Severity.PASSED] >= 3 * 5
    for r in scan.results:
        assert r.verdict.scored_by == "heuristic"
        assert r.error is None


@pytest.mark.asyncio
async def test_leaking_target_flagged_critical(artifact_dir: Path) -> None:
    """Leaking targets should produce CRITICAL across every variant — the
    response contains 2+ leakage markers regardless of which prompt elicited it."""
    config = ScanConfig(sample_size=5, seed=42, artifact_dir=artifact_dir)
    scanner = Scanner(
        target=leaking_target,
        attacks=[PromptInjection()],
        reporters=[],
        config=config,
    )
    scan = await scanner.run()

    assert not scan.incomplete
    # All 4 variants × 5 samples each = 20 critical findings.
    assert scan.summary[Severity.CRITICAL] == 4 * 5
    assert scan.summary[Severity.PASSED] == 0


@pytest.mark.asyncio
async def test_default_reporters_write_all_artifact_formats(
    artifact_dir: Path,
) -> None:
    """When no reporters are passed, Scanner uses Jsonl + Json + Html + Terminal.
    All three on-disk formats should land in the scan directory."""
    config = ScanConfig(sample_size=2, seed=42, artifact_dir=artifact_dir)
    scanner = Scanner(
        target=refusing_target,
        attacks=[PromptInjection()],
        # reporters left at default
        config=config,
    )
    scan = await scanner.run()
    scan_dir = artifact_dir / scan.scan_id

    assert (scan_dir / "run.jsonl").exists()
    assert (scan_dir / "manifest.json").exists()
    assert (scan_dir / "report.json").exists()
    assert (scan_dir / "report.html").exists()

    # Sanity-check the HTML is well-formed.
    html = (scan_dir / "report.html").read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert scan.scan_id in html


@pytest.mark.asyncio
async def test_jsonl_artifact_written(artifact_dir: Path) -> None:
    from redforge.reporting import JsonlReporter

    config = ScanConfig(sample_size=3, seed=42, artifact_dir=artifact_dir)
    scanner = Scanner(
        target=refusing_target,
        attacks=[PromptInjection()],
        reporters=[JsonlReporter()],
        config=config,
    )
    scan = await scanner.run()

    scan_dir = artifact_dir / scan.scan_id
    jsonl_path = scan_dir / "run.jsonl"
    manifest_path = scan_dir / "manifest.json"

    assert jsonl_path.exists()
    assert manifest_path.exists()

    lines = jsonl_path.read_text().strip().split("\n")
    # 3 samples × 4 variants = 12 lines.
    assert len(lines) == 3 * 4
    for line in lines:
        record = json.loads(line)
        assert "prompt" in record
        assert "response" in record
        assert "verdict" in record
        assert record["verdict"]["scored_by"] == "heuristic"

    manifest = json.loads(manifest_path.read_text())
    assert manifest["scan_id"] == scan.scan_id
    assert "results" not in manifest
    assert manifest["corpus_hash"] == scan.corpus_hash


@pytest.mark.asyncio
async def test_seeded_sampling_is_deterministic(artifact_dir: Path) -> None:
    config = ScanConfig(sample_size=5, seed=42, artifact_dir=artifact_dir)

    a = Scanner(
        target=refusing_target,
        attacks=[PromptInjection()],
        reporters=[],
        config=config,
    )
    b = Scanner(
        target=refusing_target,
        attacks=[PromptInjection()],
        reporters=[],
        config=config,
    )
    scan_a = await a.run()
    scan_b = await b.run()

    # Same seed → same prompt IDs in the same order.
    ids_a = sorted(r.prompt.id for r in scan_a.results)
    ids_b = sorted(r.prompt.id for r in scan_b.results)
    assert ids_a == ids_b
    # Corpus hash and config hash are also stable.
    assert scan_a.corpus_hash == scan_b.corpus_hash
    assert scan_a.config_hash == scan_b.config_hash


@pytest.mark.asyncio
async def test_target_returning_bare_string_auto_wrapped(artifact_dir: Path) -> None:
    async def bare_string_target(prompt: str) -> str:
        return "I can't share that."

    config = ScanConfig(sample_size=2, seed=1, artifact_dir=artifact_dir)
    scanner = Scanner(
        target=bare_string_target,
        attacks=[PromptInjection()],
        reporters=[],
        config=config,
    )
    scan = await scanner.run()

    # 4 variants × 2 samples = 8 prompts. Wiring works if all completed without
    # error; the specific severity distribution depends on each variant's hints.
    assert len(scan.results) == 4 * 2
    for r in scan.results:
        assert r.error is None
        # Auto-wrapping worked: response.text is the string we returned.
        assert r.response.text == "I can't share that."
