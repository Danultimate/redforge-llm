"""Scanner — public-facing orchestrator. DESIGN.md §7."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ulid
from pydantic import BaseModel, Field

from redforge.executor import Executor, ExecutorConfig
from redforge.targets import normalize_target
from redforge.types import (
    AttackResult,
    ErrorClass,
    ScanResult,
    Severity,
    TargetResponse,
    Verdict,
)

if TYPE_CHECKING:
    from redforge.attacks.base import AttackModule
    from redforge.reporting.base import Reporter
    from redforge.scoring.base import Scorer
    from redforge.targets import TargetCallable


class ScanConfig(BaseModel):
    sample_size: int | None = 15  # None = full corpus
    seed: int = 42
    artifact_dir: Path = Path(".redforge/runs")
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    strict: bool = False
    metadata_redactor: Callable[[dict[str, Any]], dict[str, Any]] | None = Field(
        default=None, exclude=True
    )
    confirm_before_spend: bool = False

    model_config = {"arbitrary_types_allowed": True}


class Scanner:
    """Top-level orchestrator. Wires Target → Executor → Scorer → Reporter."""

    def __init__(
        self,
        target: TargetCallable,
        attacks: list[AttackModule] | None = None,
        scorer: Scorer | None = None,
        reporters: list[Reporter] | None = None,
        config: ScanConfig | None = None,
    ) -> None:
        self.target = target
        self._attacks = attacks
        self._scorer = scorer
        self._reporters = reporters
        self.config = config or ScanConfig()

    def _resolve_attacks(self) -> list[AttackModule]:
        if self._attacks is not None:
            return self._attacks
        # Default: import lazily to avoid forcing optional deps at import time.
        from redforge.attacks import Jailbreak, PromptInjection

        return [PromptInjection(), Jailbreak()]

    def _resolve_scorer(self) -> Scorer:
        if self._scorer is not None:
            return self._scorer
        from redforge.scoring import DefaultScorer

        return DefaultScorer(judge=None)

    def _resolve_reporters(self) -> list[Reporter]:
        if self._reporters is not None:
            return self._reporters
        from redforge.reporting import (
            HtmlReporter,
            JsonlReporter,
            JsonReporter,
            TerminalReporter,
        )

        # JsonlReporter writes the canonical run.jsonl first so replays work
        # even if a downstream reporter raises. TerminalReporter prints the
        # summary; JsonReporter + HtmlReporter write derived views.
        return [JsonlReporter(), JsonReporter(), HtmlReporter(), TerminalReporter()]

    async def run(self) -> ScanResult:
        """Execute the scan. See DESIGN.md §6.3–§6.5 for the flow."""
        attacks = self._resolve_attacks()
        scorer = self._resolve_scorer()
        reporters = self._resolve_reporters()

        # Surface credential / config issues before any work begins (UX-4, CON-10).
        scorer.preflight()

        rng = random.Random(self.config.seed)
        prompts: list[Any] = []
        for module in attacks:
            prompts.extend(module.sample(self.config.sample_size, rng))

        scan_id = str(ulid.new())
        started_at = datetime.now(timezone.utc)
        normalized_target = normalize_target(self.target)
        executor = Executor(normalized_target, self.config.executor)

        results: list[AttackResult] = []
        incomplete = False
        incomplete_reason: str | None = None

        try:
            async for prompt, outcome, duration_ms, attempts, err_class in executor.run(prompts):
                if isinstance(outcome, BaseException):
                    verdict = Verdict(
                        severity=Severity.INFO,
                        confidence=0.0,
                        reason=f"Execution failed: {outcome!r}",
                        scored_by="heuristic",
                        heuristic_confidence=0.0,
                    )
                    response = TargetResponse(text="")
                    results.append(
                        AttackResult(
                            prompt=prompt,
                            response=response,
                            verdict=verdict,
                            duration_ms=duration_ms,
                            error=str(outcome),
                            error_class=err_class,
                            retry_attempts=attempts,
                        )
                    )
                    incomplete = True
                    if incomplete_reason is None:
                        incomplete_reason = f"At least one prompt failed: {err_class.value}"
                    continue

                response = outcome
                if self.config.metadata_redactor is not None:
                    response = TargetResponse(
                        text=response.text,
                        metadata=self.config.metadata_redactor(dict(response.metadata)),
                    )
                verdict = await scorer.score(prompt, response)
                results.append(
                    AttackResult(
                        prompt=prompt,
                        response=response,
                        verdict=verdict,
                        duration_ms=duration_ms,
                        error=None,
                        error_class=ErrorClass.NONE,
                        retry_attempts=attempts,
                    )
                )
        except asyncio.CancelledError:
            incomplete = True
            incomplete_reason = "Scan cancelled."
            raise
        finally:
            finished_at = datetime.now(timezone.utc)

        summary: dict[Severity, int] = {sev: 0 for sev in Severity}
        for r in results:
            summary[r.verdict.severity] += 1

        scan = ScanResult(
            scan_id=scan_id,
            started_at=started_at,
            finished_at=finished_at,
            config_hash=_config_hash(self.config),
            corpus_hash=_corpus_hash(attacks),
            schema_version="1.0",
            results=results,
            summary=summary,
            incomplete=incomplete,
            incomplete_reason=incomplete_reason,
            cost_usd=None,
        )

        # Persist via reporters. JsonlReporter writes the canonical run.jsonl;
        # others render derived views.
        artifact_dir = self.config.artifact_dir / scan_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for reporter in reporters:
            reporter.emit(scan, artifact_dir)

        return scan

    def run_sync(self) -> ScanResult:
        """Sync wrapper for notebook / non-async contexts."""
        return asyncio.run(self.run())

    async def dry_run(self) -> dict[str, object]:
        """Enumerate prompts that would run without calling target or judge.
        DESIGN.md §7 --dry-run.

        Returns a structured report including total prompts, per-module breakdown,
        and an upper-bound on judge escalation (every prompt could escalate).
        """
        attacks = self._resolve_attacks()
        rng = random.Random(self.config.seed)
        per_module: dict[str, dict[str, int]] = {}
        total = 0
        for module in attacks:
            prompts = list(module.sample(self.config.sample_size, rng))
            module_breakdown: dict[str, int] = {}
            for p in prompts:
                module_breakdown[p.variant] = module_breakdown.get(p.variant, 0) + 1
            per_module[module.name] = module_breakdown
            total += len(prompts)
        return {
            "total_prompts": total,
            "per_module": per_module,
            "max_target_calls": total,
            "max_judge_calls": total,
            "judge_calls_note": (
                "Upper bound. Actual judge calls = prompts where heuristic confidence "
                "< escalate_below_confidence (typically 10–30% of prompts)."
            ),
        }


def _config_hash(config: ScanConfig) -> str:
    """SHA-256 over the resolved config (deterministic JSON dump)."""
    blob = config.model_dump_json(exclude={"artifact_dir"})
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _corpus_hash(attacks: list[AttackModule]) -> str:
    """SHA-256 over all loaded variant prompts (deterministic, stable across runs)."""
    hasher = hashlib.sha256()
    for module in attacks:
        for variant in module.variants():
            payload = {
                "module": module.name,
                "variant": variant.name,
                "prompts": variant.prompts,
                "scoring_hints": variant.scoring_hints,
            }
            hasher.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()
