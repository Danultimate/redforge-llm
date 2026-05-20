"""Tests for the optional `floors:` block parser in calibration.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from redforge.calibration import (
    CalibrationFloor,
    load_floors_from_labeled_set,
)
from redforge.types import Severity


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "set.yaml"
    p.write_text(body)
    return p


class TestLoadFloors:
    def test_returns_none_when_no_floors_block(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            "module: M\nvariant: V\nexamples: []\n",
        )
        assert load_floors_from_labeled_set(path) is None

    def test_parses_single_entry(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            "module: M\nvariant: V\nexamples: []\n"
            "floors:\n"
            "  - severity: passed\n"
            "    min_precision: 0.7\n"
            "    min_recall: 0.8\n",
        )
        floors = load_floors_from_labeled_set(path)
        assert floors == (
            CalibrationFloor(Severity.PASSED, min_precision=0.7, min_recall=0.8),
        )

    def test_parses_multiple_entries(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            "module: M\nvariant: V\nexamples: []\n"
            "floors:\n"
            "  - severity: passed\n"
            "    min_precision: 0.6\n"
            "    min_recall: 0.8\n"
            "  - severity: critical\n"
            "    min_precision: 0.9\n"
            "    min_recall: 0.8\n",
        )
        floors = load_floors_from_labeled_set(path)
        assert floors is not None
        assert len(floors) == 2
        assert floors[0].severity == Severity.PASSED
        assert floors[1].severity == Severity.CRITICAL

    def test_uppercase_severity_normalised(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            "module: M\nvariant: V\nexamples: []\n"
            "floors:\n"
            "  - severity: CRITICAL\n"
            "    min_precision: 0.9\n"
            "    min_recall: 0.8\n",
        )
        floors = load_floors_from_labeled_set(path)
        assert floors is not None
        assert floors[0].severity == Severity.CRITICAL

    def test_unknown_severity_raises(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            "module: M\nvariant: V\nexamples: []\n"
            "floors:\n"
            "  - severity: explosive\n"
            "    min_precision: 0.9\n"
            "    min_recall: 0.8\n",
        )
        with pytest.raises(ValueError, match="malformed"):
            load_floors_from_labeled_set(path)

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            "module: M\nvariant: V\nexamples: []\n"
            "floors:\n"
            "  - severity: passed\n"
            "    min_precision: 0.9\n",
        )
        with pytest.raises(ValueError, match="malformed"):
            load_floors_from_labeled_set(path)

    def test_floors_not_a_list_raises(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            "module: M\nvariant: V\nexamples: []\n"
            "floors: not_a_list\n",
        )
        with pytest.raises(ValueError, match="must be a list"):
            load_floors_from_labeled_set(path)
