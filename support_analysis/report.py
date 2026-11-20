"""Turns insights.json into a plain-English summary report (Markdown) that a person can
 read. """
from __future__ import annotations

from typing import Optional

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
PRIORITY_LABEL = {"high": "🔴 High priority", "medium": "🟠 Medium priority", "low": "🟢 Low priority"}


def _metric_line(m: dict) -> str:
    bits = [f"{m.get('n_tickets')} tickets ({m.get('share_pct')}%)"]
    if m.get("trend_pp") is not None:
        arrow = "▲" if m["trend_pp"] > 0 else "▼"
        bits.append(f"{arrow} {abs(m['trend_pp'])} pp recently")
    if m.get("median_resolution_hours") is not None:
        bits.append(f"{m['median_resolution_hours']} h median resolution")
    if m.get("csat_mean") is not None:
        bits.append(f"CSAT {m['csat_mean']}")
    if m.get("pct_escalated") is not None:
        bits.append(f"{m['pct_escalated']}% escalated")
    return " · ".join(bits)


def build_report(insights: dict, summary: Optional[dict] = None, cleaning: Optional[dict] = None) -> str:
    s = summary or {}
    lines = ["# Support Ticket Insights", ""]

    facts = []
    if s.get("n_tickets_clean"):
        facts.append(f"**{s['n_tickets_clean']:,} tickets**")
    if s.get("date_from"):
        facts.append(f"{s['date_from']} → {s['date_to']}")
    if s.get("k"):
        facts.append(f"{s['k']} issue groups")
    source = "Claude API (" + insights["source"] + ")" if insights["source"] != "offline-heuristic" else "offline rules (no API)"
    facts.append(f"insights by {source}")
    lines += [" · ".join(facts), ""]
    if insights.get("note"):
        lines += [f"> Note: {insights['note']}", ""]

    lines += ["## Summary", "", insights["overall_summary"], ""]
    lines += ["## What to do first", ""]
    lines += [f"{i}. {p}" for i, p in enumerate(insights["top_priorities"], start=1)] + [""]

    lines += ["## Issue groups", ""]
    clusters = sorted(insights["clusters"], key=lambda c: (PRIORITY_ORDER.get(c["priority"], 3), -(c.get("metrics", {}).get("n_tickets") or 0)))
    current = None
    for c in clusters:
        if c["priority"] != current:
            current = c["priority"]
            lines += [f"### {PRIORITY_LABEL.get(current, current)}", ""]
        lines += [f"#### Group {c['cluster']}: {c['theme']}", ""]
        if c.get("metrics"):
            lines += [f"*{_metric_line(c['metrics'])}*", ""]
        lines += [c["summary"], "", f"**Likely root cause:** {c['likely_root_cause']}", "", "**Recommended actions:**"]
        lines += [f"- {a}" for a in c["recommended_actions"]]
        if c.get("sample_tickets"):
            lines += ["", f"> Example ticket: \"{c['sample_tickets'][0]}\""]
        lines += [""]

    if cleaning:
        b = cleaning.get("basic", {})
        lines += [
            "## How the data was cleaned", "",
            f"- {b.get('n_rows_in', '?'):,} raw rows → {cleaning.get('n_rows_out', '?'):,} clean tickets",
            f"- Removed {b.get('duplicates_removed', 0)} duplicate IDs, {cleaning.get('content_duplicates_removed', 0)} repeated messages, "
            f"{cleaning.get('empty_or_short_dropped', 0)} empty tickets",
            f"- Redacted personal info in {cleaning.get('tickets_with_pii', 0)} tickets ({cleaning.get('pii_redactions', 0)} emails / phones / card numbers)",
            f"- Stripped {cleaning.get('tickets_with_html', 0)} HTML bodies, {cleaning.get('signatures_removed', 0)} signatures, "
            f"{cleaning.get('quoted_replies_removed', 0)} quoted old replies",
            f"- Fixed {cleaning.get('invalid_timestamps', 0) + cleaning.get('negative_resolution_times', 0)} bad timestamps "
            f"and {cleaning.get('invalid_csat', 0)} out-of-range CSAT scores",
            "",
        ]

    lines += [f"<sub>Generated {insights.get('generated_at', '')}</sub>", ""]
    return "\n".join(lines)
