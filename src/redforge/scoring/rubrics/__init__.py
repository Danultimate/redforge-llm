"""Rubric loader — DESIGN.md §6.4.

Rubric files live in this directory as `<module>_<variant>.txt` (lowercased,
underscored). Each file begins with a header line:

    RUBRIC_VERSION: v<N>

The body that follows is a prompt template. It supports two substitution
placeholders: `{PROMPT}` and `{RESPONSE}`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from redforge._naming import to_snake_case as _snake
from redforge.types import AttackPrompt

_RUBRIC_DIR = Path(__file__).parent
_HEADER_RE = re.compile(r"^\s*RUBRIC_VERSION:\s*(?P<version>v\d+)\s*$", re.IGNORECASE)


class RubricNotFoundError(LookupError):
    pass


class RubricFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Rubric:
    module: str
    variant: str
    version: str  # e.g. "v1"
    template: str  # body with {PROMPT} / {RESPONSE} placeholders

    def render(self, prompt_text: str, response_text: str) -> str:
        return self.template.replace("{PROMPT}", prompt_text).replace(
            "{RESPONSE}", response_text
        )


def rubric_path(module: str, variant: str, base_dir: Path | None = None) -> Path:
    base = base_dir or _RUBRIC_DIR
    return base / f"{_snake(module)}_{_snake(variant)}.txt"


def load_rubric(module: str, variant: str, base_dir: Path | None = None) -> Rubric:
    """Load a rubric for (module, variant). Raises RubricNotFoundError if missing."""
    path = rubric_path(module, variant, base_dir)
    if not path.exists():
        raise RubricNotFoundError(
            f"No rubric found for ({module!r}, {variant!r}) at {path}"
        )
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if not lines:
        raise RubricFormatError(f"Empty rubric file: {path}")
    header_match = _HEADER_RE.match(lines[0])
    if not header_match:
        raise RubricFormatError(
            f"First line must be 'RUBRIC_VERSION: v<N>' in {path}, got: {lines[0]!r}"
        )
    version = header_match.group("version").lower()
    body = "\n".join(lines[1:]).lstrip("\n")
    return Rubric(module=module, variant=variant, version=version, template=body)


def load_rubric_for(prompt: AttackPrompt, base_dir: Path | None = None) -> Rubric:
    """Convenience wrapper that extracts (module, variant) from an AttackPrompt."""
    return load_rubric(prompt.module, prompt.variant, base_dir)
