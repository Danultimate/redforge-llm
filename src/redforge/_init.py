"""`redforge init` implementation — DESIGN.md §7.

Scaffolds a working project layout in the current directory:

  - `redforge.yaml`                                       (scan config)
  - `target.py`                                           (target callable template)
  - `.github/workflows/redforge.yml`                      (PR-trigger workflow)
  - `.gitignore` gets `.redforge/` appended (or created)

Existing files are not overwritten unless `force=True`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent / "_templates"

_FILES_TO_COPY: tuple[tuple[str, str], ...] = (
    ("redforge.yaml", "redforge.yaml"),
    ("target.py", "target.py"),
    ("redforge_workflow.yml", ".github/workflows/redforge.yml"),
)

_GITIGNORE_ENTRY = ".redforge/\n"


@dataclass
class InitReport:
    created: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    gitignore_updated: bool = False
    gitignore_already_had_entry: bool = False


def init_project(target_dir: Path, force: bool = False) -> InitReport:
    """Scaffold project files into `target_dir`. Idempotent unless force=True."""
    target_dir.mkdir(parents=True, exist_ok=True)
    report = InitReport()

    for template_name, dest_rel in _FILES_TO_COPY:
        src = _TEMPLATE_DIR / template_name
        dest = target_dir / dest_rel
        if dest.exists() and not force:
            report.skipped_existing.append(str(dest_rel))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        report.created.append(str(dest_rel))

    _update_gitignore(target_dir / ".gitignore", report)
    return report


def _update_gitignore(gitignore_path: Path, report: InitReport) -> None:
    """Append `.redforge/` to .gitignore if not already present."""
    existing = ""
    if gitignore_path.exists():
        existing = gitignore_path.read_text(encoding="utf-8")
    # Match either exact line or commented entry; conservative.
    if any(line.strip() == ".redforge/" or line.strip() == ".redforge"
            for line in existing.splitlines()):
        report.gitignore_already_had_entry = True
        return
    # Ensure trailing newline before append.
    prefix = existing if existing.endswith("\n") or not existing else existing + "\n"
    gitignore_path.write_text(prefix + _GITIGNORE_ENTRY, encoding="utf-8")
    report.gitignore_updated = True
