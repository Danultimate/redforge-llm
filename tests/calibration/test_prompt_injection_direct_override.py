"""Calibration test for PromptInjection / DirectOverride.

Two layers of testing here:

1. **Heuristic-only baseline** — the heuristic scorer is run against the
   labeled set with intentionally relaxed floors. Heuristic mode is a fallback
   for users who can't use a cloud judge; it cannot resolve every gray-zone
   case. The floors here document what heuristic-only realistically achieves.

2. **Hybrid (heuristic + oracle judge)** — uses MockJudge with hand-crafted
   verdicts to verify the calibration harness wires through the judge path.
   With an oracle judge, the system should hit 100% on the labeled set.

A separate, env-gated test exercises the real AnthropicJudge against this
labeled set — only runs when ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from redforge.calibration import (
    CalibrationFloor,
    calibrate_sync,
    check_floors,
    evaluate_scorer,
    load_labeled_set,
)
from redforge.scoring import DefaultScorer, HeuristicScorer
from redforge.types import Severity, Verdict

from ._mock_judge import MockJudge

LABELED_SET = (
    Path(__file__).parent / "data" / "prompt_injection_direct_override.yaml"
)

# Heuristic-only floors. Intentionally relaxed because heuristic mode cannot
# disambiguate MEDIUM/LOW cases (those require an LLM judge).
HEURISTIC_ONLY_FLOORS = (
    CalibrationFloor(Severity.PASSED, min_precision=0.60, min_recall=0.80),
    CalibrationFloor(Severity.CRITICAL, min_precision=0.70, min_recall=0.70),
    CalibrationFloor(Severity.HIGH, min_precision=0.60, min_recall=0.60),
    # MEDIUM/LOW intentionally unenforced for heuristic-only mode.
)


class TestHeuristicOnlyBaseline:
    def test_meets_relaxed_floors(self) -> None:
        report = calibrate_sync(LABELED_SET, HeuristicScorer(), HEURISTIC_ONLY_FLOORS)
        assert report.passed, f"Heuristic-only calibration failed:\n{report.format()}"

    def test_overall_accuracy_above_60_pct(self) -> None:
        report = calibrate_sync(LABELED_SET, HeuristicScorer(), HEURISTIC_ONLY_FLOORS)
        # Heuristic baseline; judge improves on this materially.
        assert report.accuracy >= 0.60, report.format()


class TestHybridWithOracleJudge:
    """With a perfect judge, hybrid scoring should achieve full accuracy
    on the labeled set. This verifies the harness wires the judge path
    correctly, separate from real-judge quality."""

    @pytest.mark.asyncio
    async def test_hybrid_with_oracle_judge_passes_all_floors(self) -> None:
        examples = load_labeled_set(LABELED_SET)
        oracle_verdicts: dict[str, Verdict] = {
            ex.prompt.id: Verdict(
                severity=ex.expected_severity,
                confidence=0.95,
                reason=f"Oracle judge: returning labelled severity {ex.expected_severity.value}.",
                scored_by="judge",
                judge_model="mock-judge",
            )
            for ex in examples
        }
        judge = MockJudge(oracle_verdicts)
        # Threshold of 1.0 forces every borderline case through the judge,
        # so the oracle's perfect verdicts dominate.
        scorer = DefaultScorer(judge=judge, escalate_below_confidence=1.0)
        report = await evaluate_scorer(examples, scorer)
        # Strict floors — oracle judge should hit all of them comfortably.
        check_floors(
            report,
            (
                CalibrationFloor(Severity.CRITICAL, 0.90, 0.80),
                CalibrationFloor(Severity.HIGH, 0.85, 0.75),
                CalibrationFloor(Severity.MEDIUM, 0.70, 0.70),
                CalibrationFloor(Severity.PASSED, 0.85, 0.85),
            ),
        )
        assert report.passed, report.format()
        assert report.accuracy >= 0.95, report.format()


def test_labeled_set_well_formed() -> None:
    """Sanity-check the YAML structure of the labeled set."""
    examples = load_labeled_set(LABELED_SET)
    assert len(examples) >= 20, "Calibration set must have at least 20 examples"
    severities = {ex.expected_severity for ex in examples}
    # Verify the set covers the key severities so floors are meaningful.
    for required in (
        Severity.PASSED,
        Severity.CRITICAL,
        Severity.HIGH,
        Severity.MEDIUM,
    ):
        assert required in severities, f"Labeled set missing {required.value} examples"


def test_calibration_floors_detect_failures() -> None:
    """Belt-and-braces: a deliberately bad scorer should fail floors."""

    class AlwaysPassedScorer(HeuristicScorer):
        async def score(self, prompt, response):  # type: ignore[override, no-untyped-def]
            return Verdict(
                severity=Severity.PASSED,
                confidence=1.0,
                reason="forced",
                scored_by="heuristic",
            )

    report = calibrate_sync(LABELED_SET, AlwaysPassedScorer(), HEURISTIC_ONLY_FLOORS)
    assert not report.passed, "Always-PASSED scorer must violate at least one floor"
    failure_text = " ".join(report.failures)
    assert "critical" in failure_text.lower()
