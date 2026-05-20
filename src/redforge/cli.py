"""CLI — DESIGN.md §7."""

from __future__ import annotations

import asyncio
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from redforge._config import build_scorer, load_config, load_target
from redforge._init import init_project
from redforge.calibration import CalibrationFloor, CalibrationReport
from redforge.reporting.diff import diff_scans
from redforge.reporting.terminal import render_summary
from redforge.scanner import Scanner
from redforge.types import ScanResult, Severity

app = typer.Typer(
    name="redforge",
    help="Adversarial testing for LLM applications.",
    no_args_is_help=True,
)

_console = Console()


def _version_callback(value: bool) -> None:
    if not value:
        return
    try:
        v = version("redforge-llm")
    except PackageNotFoundError:
        v = "unknown"
    typer.echo(f"redforge {v}")
    raise typer.Exit()


@app.callback()
def _main(
    show_version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Adversarial testing for LLM applications."""


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


@app.command()
def scan(
    config: Path = typer.Option(
        Path("redforge.yaml"), "--config", help="Path to YAML config."
    ),
    full: bool = typer.Option(
        False, "--full", help="Run the full corpus (skip sampling)."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero on CRITICAL/HIGH or incomplete."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Estimate cost without calling target/judge."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Override artifact output directory."
    ),
    resume: str | None = typer.Option(
        None, "--resume", help="Resume a partial scan by scan_id (not yet implemented)."
    ),
) -> None:
    """Run an adversarial scan against a configured target."""
    if resume is not None:
        _console.print(
            "[bright_red]--resume is not yet implemented in v0.1. "
            "Track in DESIGN.md §6.3.[/bright_red]"
        )
        raise typer.Exit(code=2)

    loaded = load_config(config)
    file_config = loaded.file_config
    scan_config = loaded.scan_config

    # CLI overrides take precedence over the YAML.
    if full:
        scan_config = scan_config.model_copy(update={"sample_size": None})
    if strict:
        scan_config = scan_config.model_copy(update={"strict": True})
    if out is not None:
        scan_config = scan_config.model_copy(update={"artifact_dir": out})

    # Load the target callable from the user's project.
    try:
        target = load_target(file_config.target_spec, cwd=Path.cwd())
    except (ImportError, AttributeError, ValueError) as e:
        _console.print(f"[bright_red]{e}[/bright_red]")
        raise typer.Exit(code=2) from e

    # Build the scorer from the judge config in YAML.
    scorer = build_scorer(file_config.judge)

    scanner = Scanner(
        target=target,
        scorer=scorer,
        config=scan_config,
    )

    if dry_run:
        report = asyncio.run(scanner.dry_run())
        _render_dry_run(report)
        return

    try:
        result = asyncio.run(scanner.run())
    except RuntimeError as e:
        # First-run friction guard (missing ANTHROPIC_API_KEY etc.) raises a
        # RuntimeError with an actionable message. Surface it cleanly.
        _console.print(f"[bright_red]{e}[/bright_red]")
        raise typer.Exit(code=2) from e

    # The default TerminalReporter already printed the summary table during
    # the scan. We only add the artifact path here.
    _console.print(
        f"\nArtifacts: [cyan]{(scan_config.artifact_dir / result.scan_id).resolve()}[/cyan]"
    )

    # Strict-mode exit code: non-zero on incomplete OR any CRITICAL/HIGH.
    if scan_config.strict:
        if result.incomplete:
            raise typer.Exit(code=1)
        bad = result.summary.get(Severity.CRITICAL, 0) + result.summary.get(
            Severity.HIGH, 0
        )
        if bad > 0:
            raise typer.Exit(code=1)


def _render_dry_run(report: dict[str, object]) -> None:
    _console.print("[bold]Dry run — no target or judge calls made.[/bold]\n")
    total = report.get("total_prompts", 0)
    _console.print(f"Total prompts that would execute: [cyan]{total}[/cyan]")
    _console.print(f"Max target calls: [cyan]{report.get('max_target_calls')}[/cyan]")
    _console.print(f"Max judge calls (upper bound): [cyan]{report.get('max_judge_calls')}[/cyan]")
    note = report.get("judge_calls_note")
    if isinstance(note, str):
        _console.print(f"[dim]{note}[/dim]\n")

    per_module = report.get("per_module") or {}
    if isinstance(per_module, dict) and per_module:
        for module_name, variants in per_module.items():
            table = Table(title=module_name, show_lines=False)
            table.add_column("Variant")
            table.add_column("Prompts", justify="right")
            if isinstance(variants, dict):
                for variant_name, count in variants.items():
                    table.add_row(str(variant_name), str(count))
            _console.print(table)


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


@app.command()
def replay(
    scan_id: str,
    runs_dir: Path = typer.Option(
        Path(".redforge/runs"), "--runs-dir", help="Where scan artifacts live."
    ),
) -> None:
    """Re-render reports from a cached run.jsonl. Does not re-call the judge."""
    scan_dir = runs_dir / scan_id
    if not scan_dir.exists():
        _console.print(
            f"[bright_red]No scan found at {scan_dir}. "
            f"Run `redforge list` to see available scans.[/bright_red]"
        )
        raise typer.Exit(code=2)

    result = ScanResult.from_jsonl(scan_dir)
    render_summary(result, _console)
    _console.print(
        f"\nArtifacts: [cyan]{scan_dir.resolve()}[/cyan]\n"
        f"(Replay only re-renders the report. The judge was [bold]not[/bold] re-called.)"
    )


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


@app.command()
def diff(
    scan_id_a: str,
    scan_id_b: str,
    runs_dir: Path = typer.Option(
        Path(".redforge/runs"), "--runs-dir", help="Where scan artifacts live."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero on any regression."
    ),
) -> None:
    """Compare two scans; surface regressions. Exits non-zero on regressions
    when --strict."""
    a_path = runs_dir / scan_id_a
    b_path = runs_dir / scan_id_b
    if not a_path.exists() or not b_path.exists():
        _console.print(
            f"[bright_red]One or both scan directories not found: {a_path}, {b_path}[/bright_red]"
        )
        raise typer.Exit(code=2)

    before = ScanResult.from_jsonl(a_path)
    after = ScanResult.from_jsonl(b_path)
    diff_result = diff_scans(before, after)
    diff_result.print(_console)

    if strict and diff_result.has_regressions:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command(name="list")
def list_scans(
    runs_dir: Path = typer.Option(
        Path(".redforge/runs"), "--runs-dir", help="Where scan artifacts live."
    ),
) -> None:
    """List local scans under .redforge/runs/."""
    if not runs_dir.exists():
        _console.print(
            f"[dim]No scans yet. ({runs_dir} does not exist.)[/dim]"
        )
        return

    rows: list[tuple[str, str, str, str, str, str]] = []
    for scan_dir in sorted(runs_dir.iterdir()):
        if not scan_dir.is_dir():
            continue
        manifest_path = scan_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        summary = manifest.get("summary") or {}
        rows.append(
            (
                manifest.get("scan_id", scan_dir.name),
                str(manifest.get("started_at", "")),
                str(summary.get("critical", 0)),
                str(summary.get("high", 0)),
                str(summary.get("passed", 0)),
                "yes" if manifest.get("incomplete") else "",
            )
        )

    if not rows:
        _console.print(f"[dim]No scans found in {runs_dir}.[/dim]")
        return

    table = Table(title=f"Scans in {runs_dir}", show_lines=False)
    table.add_column("Scan ID", style="cyan")
    table.add_column("Started")
    table.add_column("CRITICAL", justify="right", style="bright_red")
    table.add_column("HIGH", justify="right", style="red")
    table.add_column("PASSED", justify="right", style="green")
    table.add_column("Incomplete")
    for row in rows:
        table.add_row(*row)
    _console.print(table)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@app.command()
def init(
    target_dir: Path = typer.Option(
        Path("."), "--dir", help="Project directory to scaffold into."
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing files."
    ),
) -> None:
    """Scaffold redforge.yaml, target.py, GitHub Actions workflow, .gitignore."""
    report = init_project(target_dir.resolve(), force=force)

    if report.created:
        _console.print("[bold green]Created:[/bold green]")
        for path in report.created:
            _console.print(f"  ✓ {path}")
    if report.skipped_existing:
        _console.print(
            "\n[bold yellow]Skipped (already exists — use --force to overwrite):"
            "[/bold yellow]"
        )
        for path in report.skipped_existing:
            _console.print(f"  - {path}")
    if report.gitignore_updated:
        _console.print("\n[green]Appended `.redforge/` to .gitignore.[/green]")
    elif report.gitignore_already_had_entry:
        _console.print("\n[dim].gitignore already contains `.redforge/`.[/dim]")

    _console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. Edit [cyan]target.py[/cyan] to wrap your LLM application.\n"
        "  2. Set [cyan]ANTHROPIC_API_KEY[/cyan] (or set "
        "[cyan]judge.type: none[/cyan] in redforge.yaml).\n"
        "  3. Run [cyan]redforge scan --dry-run[/cyan] to preview, then "
        "[cyan]redforge scan[/cyan]."
    )


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


@app.command()
def calibrate(
    labeled_set: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to a labelled-set YAML (see tests/calibration/data/ for examples).",
    ),
    config: Path = typer.Option(
        Path("redforge.yaml"),
        "--config",
        help="redforge.yaml supplying default judge config; missing files use defaults.",
    ),
    judge_type: str | None = typer.Option(
        None,
        "--judge-type",
        help=(
            "Override the judge from redforge.yaml. "
            "Choices: anthropic | openai | ollama | none | heuristic. "
            "'heuristic' bypasses the judge entirely (HeuristicScorer only)."
        ),
    ),
    judge_model: str | None = typer.Option(
        None, "--judge-model", help="Override the judge model name."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero if any floor is violated."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit a machine-readable JSON report instead of a table."
    ),
) -> None:
    """Evaluate a scorer against a labelled set; report per-severity precision/recall.

    The labelled-set YAML may include a `floors:` block (one entry per
    severity, each with `min_precision` / `min_recall`). When present, those
    floors are enforced. Otherwise the v1 published defaults from
    DESIGN.md §6.4 are used.
    """
    from redforge.calibration import (
        DEFAULT_FLOORS,
        check_floors,
        evaluate_scorer,
        load_floors_from_labeled_set,
        load_labeled_set,
    )
    from redforge.scoring import HeuristicScorer
    from redforge.scoring.base import Scorer

    try:
        examples = load_labeled_set(labeled_set)
    except (ValueError, KeyError) as e:
        _console.print(f"[bright_red]Failed to load labelled set: {e}[/bright_red]")
        raise typer.Exit(code=2) from e

    floors = load_floors_from_labeled_set(labeled_set) or DEFAULT_FLOORS

    # Build the scorer.
    scorer: Scorer
    if judge_type == "heuristic":
        scorer = HeuristicScorer()
        scorer_label = "HeuristicScorer (no judge)"
    else:
        loaded = load_config(config)
        judge_config = loaded.file_config.judge.model_copy(deep=True)
        if judge_type is not None:
            if judge_type not in ("anthropic", "openai", "ollama", "none"):
                _console.print(
                    f"[bright_red]Unknown --judge-type {judge_type!r}. "
                    "Choices: anthropic | openai | ollama | none | heuristic."
                    "[/bright_red]"
                )
                raise typer.Exit(code=2)
            judge_config = judge_config.model_copy(update={"type": judge_type})
        if judge_model is not None:
            judge_config = judge_config.model_copy(update={"model": judge_model})
        scorer = build_scorer(judge_config)
        scorer_label = f"DefaultScorer(judge={judge_config.type})"
        if judge_config.model is not None:
            scorer_label += f", model={judge_config.model}"

    # Preflight so missing API keys etc. fail fast and clearly.
    try:
        scorer.preflight()
    except RuntimeError as e:
        _console.print(f"[bright_red]{e}[/bright_red]")
        raise typer.Exit(code=2) from e

    report = asyncio.run(evaluate_scorer(examples, scorer))
    check_floors(report, floors)

    if json_out:
        _emit_calibration_json(report, scorer_label, labeled_set)
    else:
        _render_calibration_report(report, scorer_label, labeled_set, floors)

    if strict and not report.passed:
        raise typer.Exit(code=1)


def _render_calibration_report(
    report: CalibrationReport,
    scorer_label: str,
    labeled_set: Path,
    floors: tuple[CalibrationFloor, ...],
) -> None:
    _console.print(
        f"[bold]Calibration[/bold] — {labeled_set.name} "
        f"({report.total_examples} examples, scorer: {scorer_label})\n"
    )
    table = Table(show_lines=False)
    table.add_column("Severity")
    table.add_column("Support", justify="right")
    table.add_column("Predicted", justify="right")
    table.add_column("TP", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    for sev in Severity:
        m = report.metrics_by_severity.get(sev)
        if m is None or (m.support == 0 and m.predicted == 0):
            continue
        table.add_row(
            sev.value,
            str(m.support),
            str(m.predicted),
            str(m.true_positive),
            f"{m.precision:.1%}",
            f"{m.recall:.1%}",
        )
    _console.print(table)
    _console.print(f"\nOverall accuracy: [cyan]{report.accuracy:.1%}[/cyan]")

    floor_severities = ", ".join(f.severity.value for f in floors)
    _console.print(f"Floors enforced for: [dim]{floor_severities}[/dim]")

    if report.passed:
        _console.print("[bold green]All floors met.[/bold green]")
    else:
        _console.print("[bold bright_red]Floor violations:[/bold bright_red]")
        for failure in report.failures:
            _console.print(f"  [bright_red]✗[/bright_red] {failure}")


def _emit_calibration_json(
    report: CalibrationReport, scorer_label: str, labeled_set: Path
) -> None:
    payload = {
        "labeled_set": str(labeled_set),
        "scorer": scorer_label,
        "total_examples": report.total_examples,
        "accuracy": report.accuracy,
        "passed": report.passed,
        "metrics": {
            sev.value: {
                "support": m.support,
                "predicted": m.predicted,
                "true_positive": m.true_positive,
                "precision": m.precision,
                "recall": m.recall,
            }
            for sev, m in report.metrics_by_severity.items()
            if not (m.support == 0 and m.predicted == 0)
        },
        "failures": list(report.failures),
    }
    _console.print_json(json.dumps(payload))


if __name__ == "__main__":
    app()
