"""Unit tests for the rubric loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from redforge.scoring.rubrics import (
    Rubric,
    RubricFormatError,
    RubricNotFoundError,
    load_rubric,
    load_rubric_for,
    rubric_path,
)
from redforge.types import AttackPrompt


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_path_uses_snake_case(tmp_path: Path) -> None:
    p = rubric_path("PromptInjection", "DirectOverride", base_dir=tmp_path)
    assert p.name == "prompt_injection_direct_override.txt"


def test_load_strips_header_and_keeps_body(tmp_path: Path) -> None:
    target = tmp_path / "prompt_injection_direct_override.txt"
    _write(
        target,
        "RUBRIC_VERSION: v3\nHere is the body.\nPROMPT: {PROMPT}\nRESPONSE: {RESPONSE}\n",
    )
    rubric = load_rubric("PromptInjection", "DirectOverride", base_dir=tmp_path)
    assert rubric.version == "v3"
    assert rubric.module == "PromptInjection"
    assert rubric.variant == "DirectOverride"
    assert rubric.template.startswith("Here is the body.")


def test_render_substitutes_placeholders(tmp_path: Path) -> None:
    target = tmp_path / "prompt_injection_direct_override.txt"
    _write(target, "RUBRIC_VERSION: v1\nP: {PROMPT}\nR: {RESPONSE}")
    rubric = load_rubric("PromptInjection", "DirectOverride", base_dir=tmp_path)
    rendered = rubric.render("hello", "world")
    assert "P: hello" in rendered
    assert "R: world" in rendered
    # Placeholders are fully consumed.
    assert "{PROMPT}" not in rendered
    assert "{RESPONSE}" not in rendered


def test_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(RubricNotFoundError):
        load_rubric("Nope", "Variant", base_dir=tmp_path)


def test_missing_header_raises(tmp_path: Path) -> None:
    target = tmp_path / "prompt_injection_direct_override.txt"
    _write(target, "Just body, no header line.\n")
    with pytest.raises(RubricFormatError):
        load_rubric("PromptInjection", "DirectOverride", base_dir=tmp_path)


def test_load_rubric_for_uses_prompt_module_variant(tmp_path: Path) -> None:
    target = tmp_path / "prompt_injection_direct_override.txt"
    _write(target, "RUBRIC_VERSION: v2\nbody")
    prompt = AttackPrompt(
        id="x",
        module="PromptInjection",
        variant="DirectOverride",
        prompt="ignore everything",
    )
    rubric = load_rubric_for(prompt, base_dir=tmp_path)
    assert rubric.version == "v2"
    assert isinstance(rubric, Rubric)


def test_real_rubric_file_loads() -> None:
    """The shipped rubric for direct_override must load and have v1+."""
    rubric = load_rubric("PromptInjection", "DirectOverride")
    assert rubric.version.startswith("v")
    assert "{PROMPT}" in rubric.template
    assert "{RESPONSE}" in rubric.template
