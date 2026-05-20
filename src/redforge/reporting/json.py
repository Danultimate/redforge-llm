"""JsonReporter — single-document report.json for CI consumption. DESIGN.md §6.5.

Where `JsonlReporter` writes one AttackResult per line for streaming/replay,
`JsonReporter` writes one well-formed JSON object containing the full ScanResult.
That makes it easy to feed into `jq`, post to a CI dashboard, or compare across
scans without parsing a JSONL stream.
"""

from __future__ import annotations

from pathlib import Path

from redforge.reporting.base import Reporter
from redforge.types import ScanResult


class JsonReporter(Reporter):
    def emit(self, scan: ScanResult, artifact_dir: Path) -> Path:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        report_path = artifact_dir / "report.json"
        # `indent=2` for human readability; sort_keys for diffability across runs.
        report_path.write_text(
            scan.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return report_path
