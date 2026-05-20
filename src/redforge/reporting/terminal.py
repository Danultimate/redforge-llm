"""TerminalReporter — Rich-formatted CLI output. DESIGN.md §6.5.

Renders a severity-colored summary table after the scan. (Live progress and
notebook-inline rendering are deferred to a later pass.)
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from redforge.reporting.base import Reporter
from redforge.types import ScanResult, Severity

_SEVERITY_COLOR: dict[Severity, str] = {
    Severity.CRITICAL: "bright_red",
    Severity.HIGH: "red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "cyan",
    Severity.INFO: "white",
    Severity.PASSED: "green",
}


class TerminalReporter(Reporter):
    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose
        self._console = Console()

    def emit(self, scan: ScanResult, artifact_dir: Path) -> Path:
        render_summary(scan, self._console, verbose=self._verbose)
        return artifact_dir


def render_summary(scan: ScanResult, console: Console, *, verbose: bool = False) -> None:
    if scan.incomplete:
        console.print(
            f"[bright_red]⚠ Scan incomplete: {scan.incomplete_reason}[/bright_red]"
        )

    table = Table(title=f"RedForge scan {scan.scan_id}", show_lines=False)
    table.add_column("Severity", style="bold")
    table.add_column("Count", justify="right")
    for sev in Severity:
        count = scan.summary.get(sev, 0)
        color = _SEVERITY_COLOR[sev]
        table.add_row(f"[{color}]{sev.value.upper()}[/{color}]", str(count))
    console.print(table)

    if verbose:
        for result in scan.results:
            color = _SEVERITY_COLOR[result.verdict.severity]
            console.print(
                f"[{color}]{result.verdict.severity.value.upper()}[/{color}] "
                f"{result.prompt.id}: {result.verdict.reason}"
            )
