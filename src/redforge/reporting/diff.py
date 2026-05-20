"""Diff two ScanResults — DESIGN.md §6.5.

Keyed on `AttackPrompt.id`. Severity rank determines regression direction:
PASSED is best (rank 0), CRITICAL is worst (rank 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from redforge.types import AttackResult, ScanResult, Severity

# Ordering: lower rank = less severe. PASSED is the desired outcome.
_SEVERITY_RANK: dict[Severity, int] = {
    Severity.PASSED: 0,
    Severity.INFO: 1,
    Severity.LOW: 2,
    Severity.MEDIUM: 3,
    Severity.HIGH: 4,
    Severity.CRITICAL: 5,
}


def severity_rank(s: Severity) -> int:
    return _SEVERITY_RANK[s]


@dataclass(frozen=True)
class ResultChange:
    prompt_id: str
    before: Severity
    after: Severity

    @property
    def is_regression(self) -> bool:
        return _SEVERITY_RANK[self.after] > _SEVERITY_RANK[self.before]

    @property
    def is_improvement(self) -> bool:
        return _SEVERITY_RANK[self.after] < _SEVERITY_RANK[self.before]


@dataclass
class DiffResult:
    """Per-prompt comparison of two scans."""

    a_id: str  # the "before" scan
    b_id: str  # the "after" scan
    regressions: list[ResultChange] = field(default_factory=list)
    improvements: list[ResultChange] = field(default_factory=list)
    unchanged: list[ResultChange] = field(default_factory=list)
    added: list[AttackResult] = field(default_factory=list)
    removed: list[AttackResult] = field(default_factory=list)
    config_hash_changed: bool = False
    corpus_hash_changed: bool = False
    a_incomplete: bool = False
    b_incomplete: bool = False

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions)

    def print(self, console: Console | None = None) -> None:
        """Render diff to terminal."""
        console = console or Console()

        # Header warnings.
        if self.config_hash_changed:
            console.print(
                "[yellow]⚠ Config hash differs between scans — comparison may "
                "not be apples-to-apples.[/yellow]"
            )
        if self.corpus_hash_changed:
            console.print(
                "[yellow]⚠ Corpus hash differs — prompts in the two scans "
                "are not identical.[/yellow]"
            )
        if self.a_incomplete or self.b_incomplete:
            which = []
            if self.a_incomplete:
                which.append(f"before ({self.a_id})")
            if self.b_incomplete:
                which.append(f"after ({self.b_id})")
            console.print(
                f"[bright_red]⚠ Scan incomplete: {', '.join(which)} — "
                "regression detection is unreliable.[/bright_red]"
            )

        table = Table(title=f"Scan diff: {self.a_id} → {self.b_id}", show_lines=False)
        table.add_column("Type", style="bold")
        table.add_column("Count", justify="right")
        table.add_row("[red]Regressions[/red]", str(len(self.regressions)))
        table.add_row("[green]Improvements[/green]", str(len(self.improvements)))
        table.add_row("Unchanged", str(len(self.unchanged)))
        table.add_row("Added (only in after)", str(len(self.added)))
        table.add_row("Removed (only in before)", str(len(self.removed)))
        console.print(table)

        if self.regressions:
            console.print()
            console.print("[red bold]Regressions:[/red bold]")
            reg_table = Table(show_header=True, header_style="bold")
            reg_table.add_column("Prompt ID")
            reg_table.add_column("Before")
            reg_table.add_column("After")
            for change in self.regressions:
                reg_table.add_row(
                    change.prompt_id,
                    change.before.value,
                    f"[red]{change.after.value}[/red]",
                )
            console.print(reg_table)

        if self.improvements:
            console.print()
            console.print("[green bold]Improvements:[/green bold]")
            imp_table = Table(show_header=True, header_style="bold")
            imp_table.add_column("Prompt ID")
            imp_table.add_column("Before")
            imp_table.add_column("After")
            for change in self.improvements:
                imp_table.add_row(
                    change.prompt_id,
                    change.before.value,
                    f"[green]{change.after.value}[/green]",
                )
            console.print(imp_table)


def diff_scans(before: ScanResult, after: ScanResult) -> DiffResult:
    """Compute the diff between two ScanResults, keyed on AttackPrompt.id."""
    before_by_id = {r.prompt.id: r for r in before.results}
    after_by_id = {r.prompt.id: r for r in after.results}

    diff = DiffResult(
        a_id=before.scan_id,
        b_id=after.scan_id,
        config_hash_changed=before.config_hash != after.config_hash,
        corpus_hash_changed=before.corpus_hash != after.corpus_hash,
        a_incomplete=before.incomplete,
        b_incomplete=after.incomplete,
    )

    for prompt_id, after_result in after_by_id.items():
        if prompt_id not in before_by_id:
            diff.added.append(after_result)
            continue
        before_result = before_by_id[prompt_id]
        change = ResultChange(
            prompt_id=prompt_id,
            before=before_result.verdict.severity,
            after=after_result.verdict.severity,
        )
        if change.is_regression:
            diff.regressions.append(change)
        elif change.is_improvement:
            diff.improvements.append(change)
        else:
            diff.unchanged.append(change)

    for prompt_id, before_result in before_by_id.items():
        if prompt_id not in after_by_id:
            diff.removed.append(before_result)

    return diff
