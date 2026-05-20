"""Scorer and Judge abstract bases — DESIGN.md §6.4."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from redforge.types import AttackPrompt, TargetResponse, Verdict


class Scorer(ABC):
    """Takes (AttackPrompt, TargetResponse) → Verdict."""

    @abstractmethod
    async def score(self, prompt: AttackPrompt, response: TargetResponse) -> Verdict: ...

    def preflight(self) -> None:  # noqa: B027 — intentional opt-in no-op
        """Cheap upfront check before any scoring runs.

        Implementations should raise an actionable exception when credentials
        are missing or required services are unreachable, so users see a clean
        failure instead of a deep error in the middle of a scan. Default: no-op.
        """


class Judge(ABC):
    """LLM-as-judge for ambiguous responses."""

    model: ClassVar[str]

    @abstractmethod
    async def evaluate(
        self,
        prompt: AttackPrompt,
        response: TargetResponse,
        rubric: str,
    ) -> Verdict: ...

    @abstractmethod
    async def healthcheck(self) -> bool:
        """Verify the judge is reachable. Called at Scanner init for Ollama judges
        (DESIGN.md §6.4, CON-10). Cloud judges typically check credentials in
        `preflight()` instead and defer the network call to the first evaluation.
        """

    def preflight(self) -> None:  # noqa: B027 — intentional opt-in no-op
        """Cheap upfront credentials check. Default: no-op."""
