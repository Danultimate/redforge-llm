"""Small naming utilities shared by rubric and mitigation loaders."""

from __future__ import annotations

import re

_CAMEL_TO_SNAKE = re.compile(r"(?<!^)(?=[A-Z])")


def to_snake_case(name: str) -> str:
    """CamelCase → snake_case. Used to map module/variant names to filenames."""
    return _CAMEL_TO_SNAKE.sub("_", name).lower()
