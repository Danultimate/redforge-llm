"""JsonlReporter — streams the canonical run.jsonl. DESIGN.md §6.5.

This is the source of truth — every AttackResult is appended as one line.
Replayable, diffable, never re-rendered.
"""

from __future__ import annotations

import json
from pathlib import Path

from redforge.reporting.base import Reporter
from redforge.types import ScanResult


class JsonlReporter(Reporter):
    def emit(self, scan: ScanResult, artifact_dir: Path) -> Path:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = artifact_dir / "run.jsonl"
        manifest_path = artifact_dir / "manifest.json"

        with jsonl_path.open("w", encoding="utf-8") as f:
            for result in scan.results:
                f.write(result.model_dump_json())
                f.write("\n")

        manifest = scan.model_dump(mode="json")
        manifest.pop("results", None)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)

        return jsonl_path
