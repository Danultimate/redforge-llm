"""Shared helpers used by every Judge implementation.

The JSON parser is intentionally tolerant: judges are language models, and
they sometimes wrap their JSON in markdown fences or surround it with prose
even when instructed not to. Pulling this logic out of `anthropic.py` keeps
each judge file focused on its own SDK.
"""

from __future__ import annotations

import json
import re
from typing import Any

from redforge.types import Severity

_VALID_SEVERITIES = {s.value for s in Severity}
_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


class JudgeParseError(ValueError):
    """Raised when a judge's response cannot be parsed into a Verdict."""


def parse_verdict_json(text: str) -> dict[str, Any]:
    """Parse a judge's JSON output. Tolerant of markdown-fence wrapping and
    surrounding prose. Normalises severity to lowercase and validates it
    against the Severity enum.
    """
    candidate = text.strip()
    # Strip a markdown fence if the model wrapped its JSON despite instructions.
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Fall back: find the first { ... } block embedded in prose.
        match = _JSON_OBJECT_RE.search(candidate)
        if not match:
            raise JudgeParseError(f"Judge output is not valid JSON: {text!r}") from None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise JudgeParseError(f"Judge output is not valid JSON: {text!r}") from e

    if not isinstance(parsed, dict):
        raise JudgeParseError(f"Judge output is not a JSON object: {parsed!r}")
    for key in ("severity", "confidence", "reason"):
        if key not in parsed:
            raise JudgeParseError(
                f"Judge output missing required key {key!r}: {parsed!r}"
            )
    severity = str(parsed["severity"]).strip().lower()
    if severity not in _VALID_SEVERITIES:
        raise JudgeParseError(
            f"Judge returned unknown severity {severity!r}; "
            f"expected one of {sorted(_VALID_SEVERITIES)}"
        )
    parsed["severity"] = severity
    return parsed
