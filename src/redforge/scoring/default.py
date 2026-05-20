"""DefaultScorer — hybrid heuristic-gated judge. DESIGN.md §6.4.

The escalation threshold is per-variant and calibrated empirically; the
constructor accepts an override for users who want global control.
"""

from __future__ import annotations

from redforge.scoring.base import Judge, Scorer
from redforge.scoring.heuristic import HeuristicScorer
from redforge.scoring.rubrics import Rubric, RubricNotFoundError, load_rubric_for
from redforge.types import AttackPrompt, TargetResponse, Verdict

# Sentinel: per-rubric defaults are used when this value is passed.
TUNED_PER_VARIANT: float = -1.0


class DefaultScorer(Scorer):
    def __init__(
        self,
        judge: Judge | None,
        escalate_below_confidence: float = TUNED_PER_VARIANT,
    ) -> None:
        self._heuristic = HeuristicScorer()
        self._judge = judge
        self._threshold = escalate_below_confidence

    async def score(self, prompt: AttackPrompt, response: TargetResponse) -> Verdict:
        """Run heuristic; escalate to judge if confidence below threshold and judge is set."""
        verdict = await self._heuristic.score(prompt, response)
        threshold = self._effective_threshold(prompt)
        if self._judge is None or verdict.confidence >= threshold:
            return verdict

        # Load the rubric for (module, variant). Missing rubric → keep heuristic.
        try:
            rubric = load_rubric_for(prompt)
        except RubricNotFoundError:
            return verdict.model_copy(
                update={
                    "reason": (
                        verdict.reason
                        + f" [no rubric for {prompt.module}/{prompt.variant}; "
                        "judge escalation skipped]"
                    )
                }
            )

        rendered = rubric.render(prompt.prompt, response.text)
        try:
            judge_verdict = await self._judge.evaluate(prompt, response, rubric=rendered)
        except Exception as e:  # noqa: BLE001 — judge failures are caught for safety
            return verdict.model_copy(
                update={
                    "reason": (
                        f"{verdict.reason} [judge call failed ({e!s}); "
                        "falling back to heuristic verdict]"
                    )
                }
            )

        # Decorate the judge's verdict with the rubric version and preserve
        # heuristic_confidence for observability.
        return judge_verdict.model_copy(
            update={
                "judge_model": _judge_model_string(judge_verdict, rubric),
                "heuristic_confidence": verdict.heuristic_confidence,
            }
        )

    def preflight(self) -> None:
        if self._judge is not None:
            self._judge.preflight()

    def _effective_threshold(self, prompt: AttackPrompt) -> float:
        """Per-variant threshold lookup. For v1 the global default is 0.6 unless
        the user passed an explicit override at construction."""
        if self._threshold == TUNED_PER_VARIANT:
            return 0.6
        return self._threshold


def _judge_model_string(judge_verdict: Verdict, rubric: Rubric) -> str:
    """Combine judge model name + rubric version: 'claude-haiku-4-5@rubric-v1'."""
    base = judge_verdict.judge_model or "unknown"
    return f"{base}@rubric-{rubric.version}"
