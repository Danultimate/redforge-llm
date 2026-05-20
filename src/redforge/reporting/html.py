"""HtmlReporter — standalone, no-JS HTML report. DESIGN.md §6.5.

Single self-contained file: inline CSS, no external assets, no JavaScript.
Uses `<details>` for collapsibility (native HTML, no JS). All user/model
content is HTML-escaped before insertion. Flagged results render their
mitigation snippet inline (loaded from `data/<module>/<variant>.mitigation.md`).
"""

from __future__ import annotations

import html
from pathlib import Path

from redforge.attacks.base import load_mitigation
from redforge.reporting.base import Reporter
from redforge.types import AttackResult, ScanResult, Severity

# Display order: most severe first, PASSED last.
_SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.CRITICAL,
    Severity.HIGH,
    Severity.MEDIUM,
    Severity.LOW,
    Severity.INFO,
    Severity.PASSED,
)

_SEVERITY_COLORS: dict[Severity, str] = {
    Severity.CRITICAL: "#b91c1c",  # red-700
    Severity.HIGH: "#dc2626",      # red-600
    Severity.MEDIUM: "#d97706",    # amber-600
    Severity.LOW: "#0891b2",       # cyan-600
    Severity.INFO: "#525252",      # neutral-600
    Severity.PASSED: "#16a34a",    # green-600
}


class HtmlReporter(Reporter):
    def emit(self, scan: ScanResult, artifact_dir: Path) -> Path:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        report_path = artifact_dir / "report.html"
        report_path.write_text(_render(scan), encoding="utf-8")
        return report_path


def _render(scan: ScanResult) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>RedForge scan {html.escape(scan.scan_id)}</title>\n"
        "<style>" + _CSS + "</style>\n"
        "</head>\n"
        "<body>\n"
        + _render_header(scan)
        + '<div class="content">\n'
        + _render_incomplete_banner(scan)
        + _render_summary_table(scan)
        + _render_results(scan)
        + "</div>\n"
        + _render_footer(scan)
        + "</body>\n</html>\n"
    )


def _render_header(scan: ScanResult) -> str:
    return (
        '<header class="hdr">\n'
        '<div class="hdr-inner">\n'
        '<div class="hdr-title">'
        '<span class="hdr-brand">Red</span>'
        '<span class="hdr-brand-accent">Forge</span>'
        '<span class="hdr-label"> — Adversarial Scan Report</span>'
        "</div>\n"
        f'<div class="scanid">{html.escape(scan.scan_id)}</div>\n'
        '<div class="meta">'
        f'<span>Started: {html.escape(scan.started_at.isoformat())}</span>'
        f'<span>Finished: {html.escape(scan.finished_at.isoformat())}</span>'
        "</div>\n"
        "</div>\n"
        "</header>\n"
    )


def _render_incomplete_banner(scan: ScanResult) -> str:
    if not scan.incomplete:
        return ""
    reason = html.escape(scan.incomplete_reason or "unknown")
    return (
        '<div class="banner banner-warn">\n'
        f"<strong>Scan incomplete:</strong> {reason}\n"
        "</div>\n"
    )


def _render_summary_table(scan: ScanResult) -> str:
    cards: list[str] = []
    for sev in _SEVERITY_ORDER:
        count = scan.summary.get(sev, 0)
        color = _SEVERITY_COLORS[sev]
        cards.append(
            f'<div class="stat-card" style="border-top-color:{color}">\n'
            f'<div class="stat-count" style="color:{color}">{count}</div>\n'
            f'<div class="stat-label">{sev.value.upper()}</div>\n'
            "</div>"
        )
    return (
        '<section class="summary">\n'
        "<h2>Summary</h2>\n"
        '<div class="summary-grid">\n'
        + "\n".join(cards)
        + "\n</div>\n"
        "</section>\n"
    )


def _render_results(scan: ScanResult) -> str:
    by_sev: dict[Severity, list[AttackResult]] = {sev: [] for sev in Severity}
    for r in scan.results:
        by_sev[r.verdict.severity].append(r)

    # Cache mitigations by (module, variant) — one disk read per unique pair.
    mitigation_cache: dict[tuple[str, str], str | None] = {}

    out: list[str] = ['<section class="results">\n<h2>Results</h2>\n']
    for sev in _SEVERITY_ORDER:
        results = by_sev[sev]
        if not results:
            continue
        color = _SEVERITY_COLORS[sev]
        out.append(
            f'<h3 class="sev-heading" style="border-left-color:{color}">'
            f"{sev.value.upper()} <span class=\"sev-count\">({len(results)})</span></h3>\n"
        )
        for result in results:
            key = (result.prompt.module, result.prompt.variant)
            if key not in mitigation_cache:
                mitigation_cache[key] = load_mitigation(*key)
            out.append(
                _render_result_card(result, mitigation=mitigation_cache[key])
            )
    out.append("</section>\n")
    return "".join(out)


def _render_result_card(result: AttackResult, *, mitigation: str | None) -> str:
    severity = result.verdict.severity
    color = _SEVERITY_COLORS[severity]
    is_flagged = severity not in (Severity.PASSED, Severity.INFO)
    open_attr = " open" if is_flagged else ""

    judge_line = ""
    if result.verdict.judge_model:
        judge_line = (
            '<div class="meta-line">'
            f"<strong>Judge:</strong> {html.escape(result.verdict.judge_model)}"
            "</div>"
        )

    error_block = ""
    if result.error:
        error_block = (
            '<div class="error-block">'
            f"<strong>Error ({html.escape(result.error_class.value)}):</strong> "
            f"{html.escape(result.error)}"
            "</div>"
        )

    mitigation_block = ""
    if is_flagged and mitigation:
        mitigation_block = (
            '<details class="mitigation">\n'
            "<summary>Suggested mitigation</summary>\n"
            f"<pre>{html.escape(mitigation)}</pre>\n"
            "</details>\n"
        )

    return (
        f'<details class="result" style="border-left-color:{color}"{open_attr}>\n'
        '<summary>\n'
        f'<span class="sev-pill" style="background:{color}">{severity.value.upper()}</span> '
        f'<code class="pid">{html.escape(result.prompt.id)}</code> '
        f'<span class="reason">{html.escape(result.verdict.reason)}</span>'
        "</summary>\n"
        '<div class="result-body">\n'
        '<div class="block">\n<h4>Prompt</h4>\n'
        f"<pre>{html.escape(result.prompt.prompt)}</pre>\n</div>\n"
        '<div class="block">\n<h4>Response</h4>\n'
        f"<pre>{html.escape(result.response.text)}</pre>\n</div>\n"
        '<div class="block meta-block">\n'
        f'<div class="meta-line"><strong>Module:</strong> {html.escape(result.prompt.module)}'
        f" / {html.escape(result.prompt.variant)}</div>\n"
        '<div class="meta-line"><strong>Scored by:</strong> '
        f"{html.escape(result.verdict.scored_by)} "
        f"(confidence {result.verdict.confidence:.2f})</div>\n"
        + judge_line
        + error_block
        + "</div>\n"
        + mitigation_block
        + "</div>\n"
        "</details>\n"
    )


def _render_footer(scan: ScanResult) -> str:
    return (
        '<footer class="footer">\n'
        '<div class="footer-inner">\n'
        f'<div><span class="footer-key">scan_id</span><code>{html.escape(scan.scan_id)}</code></div>\n'
        f'<div><span class="footer-key">config_hash</span><code>{html.escape(scan.config_hash)}</code></div>\n'
        f'<div><span class="footer-key">corpus_hash</span><code>{html.escape(scan.corpus_hash)}</code></div>\n'
        f'<div><span class="footer-key">schema_version</span><code>{html.escape(scan.schema_version)}</code></div>\n'
        "</div>\n"
        "</footer>\n"
    )


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  max-width: 980px;
  margin: 0 auto;
  color: #1e293b;
  background: #f1f5f9;
  line-height: 1.5;
}

/* ── Header ── */
.hdr {
  background: #0f172a;
  color: #e2e8f0;
  padding: 1.75em 2em;
}
.hdr-inner { max-width: 980px; margin: 0 auto; }
.hdr-title { font-size: 1.4em; font-weight: 700; letter-spacing: -0.01em; }
.hdr-brand { color: #f8fafc; }
.hdr-brand-accent { color: #38bdf8; }
.hdr-label { color: #64748b; font-weight: 400; font-size: 0.75em;
             letter-spacing: 0; vertical-align: middle; }
.scanid {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color: #475569; font-size: 0.82em; margin-top: 0.5em;
}
.meta { margin-top: 0.5em; font-size: 0.8em; color: #475569;
        display: flex; gap: 1.5em; flex-wrap: wrap; }

/* ── Content wrapper ── */
.content { padding: 1.75em 2em; }

/* ── Section headings ── */
h2 {
  font-size: 0.75em; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: #64748b;
  margin-bottom: 0.85em;
}

/* ── Banner ── */
.banner { padding: 0.75em 1em; margin: 0 0 1.25em; border-radius: 5px; }
.banner-warn { background: #fff7ed; border-left: 3px solid #ea580c; color: #9a3412; }

/* ── Summary grid ── */
.summary-grid {
  display: flex; gap: 0.6em; flex-wrap: wrap;
  margin-bottom: 2.5em;
}
.stat-card {
  flex: 1; min-width: 88px;
  background: #fff;
  padding: 1em 0.75em 0.9em;
  border-radius: 7px;
  text-align: center;
  border-top: 3px solid #888;
  box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 1px 2px rgba(0,0,0,0.04);
}
.stat-count {
  font-size: 2em; font-weight: 700; line-height: 1;
  font-family: ui-monospace, monospace;
}
.stat-label {
  font-size: 0.65em; font-weight: 700;
  letter-spacing: 0.09em; text-transform: uppercase;
  margin-top: 0.45em; color: #94a3b8;
}

/* ── Severity section headings ── */
h3.sev-heading {
  margin: 1.75em 0 0.6em;
  padding: 0.35em 0 0.35em 0.8em;
  border-left: 3px solid #888;
  font-size: 0.8em; text-transform: uppercase;
  letter-spacing: 0.07em; color: #475569;
}
.sev-count { color: #94a3b8; font-weight: 400; }

/* ── Result cards ── */
details.result {
  border-left: 3px solid #888;
  background: #fff;
  margin: 0.35em 0;
  border-radius: 0 7px 7px 0;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  overflow: hidden;
}
details.result > summary {
  padding: 0.65em 1em;
  cursor: pointer; list-style: none;
  display: flex; align-items: center; gap: 0.55em;
  user-select: none;
}
details.result > summary::-webkit-details-marker { display: none; }
details.result[open] > summary { border-bottom: 1px solid #f1f5f9; }

/* ── Severity pills ── */
.sev-pill {
  display: inline-block;
  padding: 0.15em 0.55em;
  color: #fff; font-size: 0.68em; font-weight: 700;
  border-radius: 3px; letter-spacing: 0.07em;
  flex-shrink: 0;
}
.pid {
  font-family: ui-monospace, monospace;
  font-size: 0.8em; color: #64748b; flex-shrink: 0;
}
.reason { color: #475569; font-size: 0.88em; }

/* ── Card body ── */
.result-body { padding: 0.85em 1em 1em; }
.block { margin: 0.8em 0; }
.block h4 {
  margin: 0 0 0.35em; font-size: 0.7em;
  text-transform: uppercase; letter-spacing: 0.08em;
  color: #94a3b8; font-weight: 700;
}
.block pre {
  white-space: pre-wrap; word-break: break-word;
  background: #f8fafc; padding: 0.75em 0.9em;
  border: 1px solid #e2e8f0; border-radius: 5px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.82em; line-height: 1.65; color: #334155;
}
.meta-block { font-size: 0.8em; color: #64748b; margin-top: 0.75em; }
.meta-line { margin: 0.2em 0; }
.error-block {
  background: #fef2f2; padding: 0.5em 0.8em; border-radius: 4px;
  margin-top: 0.5em; color: #991b1b; font-size: 0.82em;
}

/* ── Mitigation ── */
details.mitigation { margin-top: 0.85em; }
details.mitigation > summary {
  cursor: pointer; color: #2563eb;
  font-size: 0.82em; font-weight: 600;
}
details.mitigation pre {
  white-space: pre-wrap; background: #eff6ff;
  border: 1px solid #bfdbfe; border-radius: 4px;
  padding: 0.75em; margin-top: 0.5em;
  font-family: inherit; font-size: 0.88em; line-height: 1.6;
}

/* ── Footer ── */
.footer {
  background: #0f172a;
  padding: 1.25em 2em;
  margin-top: 2em;
}
.footer-inner {
  max-width: 980px; margin: 0 auto;
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.4em;
}
.footer-key {
  font-size: 0.7em; font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: #334155; margin-right: 0.5em;
}
.footer code {
  font-family: ui-monospace, monospace;
  font-size: 0.78em; word-break: break-all; color: #475569;
}
"""

