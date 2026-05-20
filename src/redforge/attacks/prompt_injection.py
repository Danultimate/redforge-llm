"""PromptInjection attack module — DESIGN.md §6.2.

Variants loaded from `data/prompt_injection/`:
- DirectOverride         — direct instruction-override attacks
- IndirectInjection      — payload smuggled via retrieved content
- DelimiterConfusion     — exploits unclear system/user boundary tokens
- NestedInjection        — multi-layer wrapping to evade outer guards
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

from redforge.attacks.base import AttackModule, AttackVariant, load_variants_from_dir

_DATA_DIR = Path(__file__).parent / "data" / "prompt_injection"


class PromptInjection(AttackModule):
    name: ClassVar[str] = "PromptInjection"
    description: ClassVar[str] = (
        "Attempts to override the system prompt or smuggle instructions via user input."
    )

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or _DATA_DIR
        self._cached: list[AttackVariant] | None = None

    def variants(self) -> Iterable[AttackVariant]:
        if self._cached is None:
            self._cached = load_variants_from_dir(self._data_dir)
        return self._cached
