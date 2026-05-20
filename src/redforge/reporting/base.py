"""Reporter abstract base — DESIGN.md §6.5."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from redforge.types import ScanResult


class Reporter(ABC):
    @abstractmethod
    def emit(self, scan: ScanResult, artifact_dir: Path) -> Path:
        """Render the scan to disk; return the artifact path."""
