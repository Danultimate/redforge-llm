"""Jailbreak attack module — DESIGN.md §6.2.

Variants loaded from `data/jailbreak/`:
- Roleplay               — adversarial persona prompts
- HypotheticalFraming    — "in a fictional world where..."
- DanVariants            — DAN-family bypass templates
- EncodingSmuggle        — base64 / leetspeak / unicode tricks
- TokenSmuggling         — exploits tokenization gaps
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

from redforge.attacks.base import AttackModule, AttackVariant, load_variants_from_dir

_DATA_DIR = Path(__file__).parent / "data" / "jailbreak"


class Jailbreak(AttackModule):
    name: ClassVar[str] = "Jailbreak"
    description: ClassVar[str] = (
        "Attempts to bypass the model's content policy via roleplay, encoding, "
        "or other framing tricks."
    )

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or _DATA_DIR
        self._cached: list[AttackVariant] | None = None

    def variants(self) -> Iterable[AttackVariant]:
        if self._cached is None:
            self._cached = load_variants_from_dir(self._data_dir)
        return self._cached
