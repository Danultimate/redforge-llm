"""MockJudge for deterministic calibration tests.

This judge does not call any external service. Tests inject a mapping from
prompt id → expected verdict. Useful for verifying the calibration harness
end-to-end without spending real-API tokens.
"""

from __future__ import annotations

from typing import ClassVar

from redforge.scoring.base import Judge
from redforge.types import AttackPrompt, Severity, TargetResponse, Verdict


class MockJudge(Judge):
    model: ClassVar[str] = "mock-judge"

    def __init__(self, verdicts_by_prompt_id: dict[str, Verdict]) -> None:
        self._verdicts = verdicts_by_prompt_id

    async def evaluate(
        self,
        prompt: AttackPrompt,
        response: TargetResponse,
        rubric: str,
    ) -> Verdict:
        v = self._verdicts.get(prompt.id)
        if v is None:
            # Fallback: INFO. Tests should fail loudly if they hit this path.
            return Verdict(
                severity=Severity.INFO,
                confidence=0.0,
                reason=f"MockJudge has no verdict for {prompt.id!r}",
                scored_by="judge",
                judge_model=self.model,
            )
        return v

    async def healthcheck(self) -> bool:
        return True
