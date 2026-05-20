"""Target layer — DESIGN.md §6.1."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from redforge.targets.helpers import from_anthropic, from_ollama, from_openai
from redforge.types import TargetResponse

TargetCallable = Callable[[str], Awaitable[TargetResponse | str]]


@runtime_checkable
class Target(Protocol):
    async def __call__(self, prompt: str) -> TargetResponse | str: ...


def normalize_target(t: TargetCallable) -> Callable[[str], Awaitable[TargetResponse]]:
    """Wrap a user callable so it always returns TargetResponse. DESIGN.md §6.1."""

    async def wrapped(prompt: str) -> TargetResponse:
        result = await t(prompt)
        if isinstance(result, str):
            return TargetResponse(text=result)
        return result

    return wrapped


__all__ = [
    "Target",
    "TargetCallable",
    "normalize_target",
    "from_anthropic",
    "from_openai",
    "from_ollama",
]
