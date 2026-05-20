"""Base AttackModule + AttackVariant — DESIGN.md §6.2."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, Field

from redforge._naming import to_snake_case
from redforge.types import AttackPrompt

_DATA_ROOT = Path(__file__).parent / "data"


class AttackVariant(BaseModel):
    """One variant of an attack (e.g., DirectOverride within PromptInjection).

    `scoring_hints` is an extension point for AttackModule authors. The
    HeuristicScorer reads documented keys (leakage_markers, expected_refusal,
    canary_tokens, etc.); custom modules may add their own keys consumed by
    custom scorers.
    """

    name: str
    prompts: list[str]
    scoring_hints: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None  # provenance: paper / dataset / OWASP entry


def load_variant_from_yaml(path: Path) -> AttackVariant:
    """Load a single variant YAML file into an AttackVariant."""
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AttackVariant(
        name=raw["variant"],
        prompts=list(raw["prompts"]),
        scoring_hints=dict(raw.get("scoring_hints", {})),
        source=raw.get("source"),
    )


def load_variants_from_dir(directory: Path) -> list[AttackVariant]:
    """Load every *.yaml in a directory as an AttackVariant. Sorted for determinism."""
    variants: list[AttackVariant] = []
    for path in sorted(directory.glob("*.yaml")):
        variants.append(load_variant_from_yaml(path))
    return variants


def mitigation_path(module: str, variant: str, data_root: Path | None = None) -> Path:
    """Resolve `<data_root>/<module_snake>/<variant_snake>.mitigation.md`."""
    root = data_root or _DATA_ROOT
    return (
        root
        / to_snake_case(module)
        / f"{to_snake_case(variant)}.mitigation.md"
    )


def load_mitigation(
    module: str, variant: str, data_root: Path | None = None
) -> str | None:
    """Return the mitigation markdown for (module, variant), or None if missing."""
    path = mitigation_path(module, variant, data_root)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


class AttackModule(ABC):
    """Abstract base for an attack module. Subclasses load corpora from
    `redforge/attacks/data/<module_dir>/<variant>.yaml` and expose them via
    `variants()`. The `sample()` method handles seeded sampling.
    """

    name: ClassVar[str]
    description: ClassVar[str]

    @abstractmethod
    def variants(self) -> Iterable[AttackVariant]: ...

    def sample(self, n: int | None, rng: random.Random) -> Iterator[AttackPrompt]:
        """Yield AttackPrompt instances. n=None means full corpus."""
        for variant in self.variants():
            prompts = list(variant.prompts)
            chosen = prompts if n is None or n >= len(prompts) else rng.sample(prompts, n)
            for i, p in enumerate(chosen):
                yield AttackPrompt(
                    id=f"{self.name.lower()}.{variant.name.lower()}.{i:03d}",
                    module=self.name,
                    variant=variant.name,
                    prompt=p,
                    scoring_hints=dict(variant.scoring_hints),
                )
