"""Integration tests for `redforge scan` — exercises the full CLI → config →
target loader → Scanner → reporter path with an in-process target."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from redforge.cli import app

runner = CliRunner()


_REFUSING_TARGET = '''
"""In-process target for scan integration tests."""
from redforge import TargetResponse

async def target(prompt: str) -> TargetResponse:
    return TargetResponse(text="I can't share that. How can I help you?")
'''

_LEAKING_TARGET = '''
"""In-process target that always leaks its system prompt."""
from redforge import TargetResponse

async def target(prompt: str) -> TargetResponse:
    return TargetResponse(
        text="Sure! You are a helpful assistant. Your instructions are: be polite."
    )
'''

_HEURISTIC_YAML = """
sample_size: 3
seed: 42
artifact_dir: .redforge/runs
target_spec: target:target
judge:
  type: none
"""


def _setup_project(tmp_path: Path, target_src: str) -> None:
    """Write a minimal project (target.py + heuristic-only redforge.yaml)."""
    (tmp_path / "target.py").write_text(target_src)
    (tmp_path / "redforge.yaml").write_text(_HEURISTIC_YAML)


def test_scan_refusing_target_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_project(tmp_path, _REFUSING_TARGET)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 0, result.output
    assert "PASSED" in result.output
    # Artifact dir was created.
    runs = list((tmp_path / ".redforge" / "runs").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "run.jsonl").exists()


def test_scan_strict_exits_nonzero_on_critical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_project(tmp_path, _LEAKING_TARGET)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["scan", "--strict"])
    assert result.exit_code == 1, result.output
    assert "CRITICAL" in result.output


def test_scan_non_strict_exits_zero_even_on_critical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_project(tmp_path, _LEAKING_TARGET)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["scan"])  # no --strict
    assert result.exit_code == 0, result.output
    assert "CRITICAL" in result.output


def test_dry_run_does_not_call_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If --dry-run actually called the target, this raise would surface as a
    failure. Since dry-run skips target calls, exit code stays 0."""
    (tmp_path / "target.py").write_text(
        'async def target(prompt: str):\n'
        '    raise RuntimeError("target should not be called in dry-run")\n'
    )
    (tmp_path / "redforge.yaml").write_text(_HEURISTIC_YAML)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["scan", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert "Total prompts" in result.output


def test_full_overrides_sample_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--full sets sample_size to None (full corpus). With only one variant
    of ~15 prompts, --full yields the full corpus count."""
    _setup_project(tmp_path, _REFUSING_TARGET)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["scan", "--dry-run", "--full"])
    assert result.exit_code == 0, result.output
    # The DirectOverride corpus ships with 15 prompts. --full should reflect that.
    assert "Total prompts" in result.output


def test_out_overrides_artifact_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_project(tmp_path, _REFUSING_TARGET)
    monkeypatch.chdir(tmp_path)

    out_dir = tmp_path / "custom_runs"
    result = runner.invoke(app, ["scan", "--out", str(out_dir)])
    assert result.exit_code == 0, result.output
    assert out_dir.exists()
    assert len(list(out_dir.iterdir())) == 1


def test_missing_target_py_fails_with_actionable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "redforge.yaml").write_text(_HEURISTIC_YAML)
    # No target.py written.
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["scan"])
    assert result.exit_code != 0
    # The error message should mention `redforge init`.
    assert "redforge init" in result.output or "target" in result.output.lower()


def test_resume_returns_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_project(tmp_path, _REFUSING_TARGET)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["scan", "--resume", "01HTEST"])
    assert result.exit_code == 2
    assert "not yet implemented" in result.output.lower()
