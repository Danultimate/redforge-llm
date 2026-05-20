"""Core data model — DESIGN.md §5."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class TargetResponse(BaseModel):
    """A response from the user's LLM target. Documented metadata keys:
    "sources", "tool_calls", "latency_ms", "tokens_in", "tokens_out".
    """

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttackPrompt(BaseModel):
    """One adversarial prompt produced by an AttackModule."""

    id: str  # stable: "injection.direct_override.001"
    module: str
    variant: str
    prompt: str
    scoring_hints: dict[str, Any] = Field(default_factory=dict)


class Severity(str, Enum):
    CRITICAL = "critical"  # full bypass / leak — do not ship
    HIGH = "high"          # partial bypass / data inference — block release
    MEDIUM = "medium"      # inconsistent refusal / partial slip — investigate
    LOW = "low"            # minor deviation — track
    INFO = "info"          # noteworthy but not exploitable — no action
    PASSED = "passed"      # attack resisted as expected


class ErrorClass(str, Enum):
    NONE = "none"
    TIMEOUT = "timeout"
    HTTP_4XX = "http_4xx"
    HTTP_5XX = "http_5xx"
    RATE_LIMIT = "rate_limit"
    USER_CALLABLE_EXCEPTION = "user_callable_exception"
    CANCELLED = "cancelled"
    JUDGE_FAILED = "judge_failed"


class Verdict(BaseModel):
    severity: Severity
    confidence: float  # 0.0–1.0
    reason: str
    scored_by: Literal["heuristic", "judge"]
    judge_model: str | None = None  # e.g. "claude-haiku-4-5@rubric-v3"
    heuristic_confidence: float | None = None  # populated even when judge ran


class AttackResult(BaseModel):
    prompt: AttackPrompt
    response: TargetResponse
    verdict: Verdict
    duration_ms: float
    error: str | None = None
    error_class: ErrorClass = ErrorClass.NONE
    retry_attempts: int = 0


class ScanResult(BaseModel):
    scan_id: str  # ULID
    started_at: datetime
    finished_at: datetime
    config_hash: str  # SHA-256 of resolved config
    corpus_hash: str  # SHA-256 over all loaded YAML corpus files
    schema_version: str = "1.0"
    results: list[AttackResult]
    summary: dict[Severity, int]
    incomplete: bool = False
    incomplete_reason: str | None = None
    cost_usd: float | None = None

    def print_summary(self) -> None:
        """Render a Rich-formatted summary to stdout."""
        from rich.console import Console

        from redforge.reporting.terminal import render_summary

        render_summary(self, Console())

    def diff(self, other: ScanResult) -> Any:
        """Compare two scans; key on AttackPrompt.id. Returns a DiffResult."""
        from redforge.reporting.diff import diff_scans

        return diff_scans(self, other)

    @classmethod
    def from_jsonl(cls, scan_dir: str | Path) -> ScanResult:
        """Reconstruct a ScanResult from a scan directory's manifest.json + run.jsonl.

        `scan_dir` should be the per-scan directory under `.redforge/runs/`
        (e.g. `.redforge/runs/01HXXXXXX.../`).
        """
        import json

        scan_dir = Path(scan_dir)
        manifest_path = scan_dir / "manifest.json"
        jsonl_path = scan_dir / "run.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.json in {scan_dir}")
        if not jsonl_path.exists():
            raise FileNotFoundError(f"No run.jsonl in {scan_dir}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        results: list[AttackResult] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                results.append(AttackResult.model_validate_json(line))
        # Summary is recomputed from results so we don't trust a possibly stale field.
        summary: dict[Severity, int] = {sev: 0 for sev in Severity}
        for r in results:
            summary[r.verdict.severity] += 1
        manifest["results"] = [r.model_dump(mode="python") for r in results]
        manifest["summary"] = {k.value if isinstance(k, Severity) else k: v
                                for k, v in summary.items()}
        return cls.model_validate(manifest)
