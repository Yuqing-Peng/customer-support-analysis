"""The cloud API step. Sends each group's numbers and a few example tickets to Claude
and gets back what the issue is, why it's happening, and what to do about it.
No API key or no internet? It falls back to simple rules so the demo still works."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Callable, Dict, List, Literal, Optional

import pandas as pd

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are a customer support operations analyst.

You receive clusters of support tickets produced by K-means on the ticket text. For each cluster you get its metrics and a few representative ticket excerpts (personal data already redacted).

For every cluster:
- name the issue theme in a few words a support lead would recognise;
- summarise what customers are reporting, citing the metrics that matter;
- give the most likely root cause, and say plainly when the excerpts are too mixed to tell;
- recommend 2-4 concrete actions for the support or product team (macros, help-center articles, routing, bug escalation, staffing);
- set a priority (high / medium / low) based on customer impact: volume, recent growth (trend_pp), low CSAT, slow resolution and escalation rate.

Then write a short overall summary and list the top priorities across all clusters. Base every statement on the data provided."""


def _clean_number(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def build_cluster_payload(profiles: pd.DataFrame, samples: Dict[int, List[str]]) -> List[dict]:
    payload = []
    for rec in profiles.to_dict(orient="records"):
        rec = {k: _clean_number(v) for k, v in rec.items()}
        rec["sample_tickets"] = samples.get(int(rec["cluster"]), [])
        payload.append(rec)
    return payload


#  Claude API

try:  
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    BaseModel = None

if BaseModel is not None:

    class ClusterInsight(BaseModel):
        cluster: int
        theme: str
        summary: str
        likely_root_cause: str
        recommended_actions: List[str]
        priority: Literal["high", "medium", "low"]

    class InsightReport(BaseModel):
        overall_summary: str
        top_priorities: List[str]
        clusters: List[ClusterInsight]


def _call_claude(payload: List[dict], context: dict, model: str) -> dict:
    import anthropic

    user_content = (
        "Dataset context:\n" + json.dumps(context, indent=2, default=str)
        + "\n\nTicket clusters:\n" + json.dumps(payload, indent=2, default=str)
    )

    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY or an `ant auth login` profile
    response = client.beta.messages.parse(
        model=model,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default", 
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        output_format=InsightReport,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError(f"Claude returned no structured report (stop_reason={response.stop_reason})")
    return {"source": response.model, **response.parsed_output.model_dump()}


#  Offline fallback

def _offline_report(payload: List[dict]) -> dict:
    """Rule-based insights so the pipeline works without network access or credentials."""
    med_res = pd.Series([p.get("median_resolution_hours") for p in payload], dtype="float").median()

    def impact(p: dict) -> float:
        score = p.get("share_pct", 0) / 10
        score += max(p.get("trend_pp") or 0, 0)
        if p.get("csat_mean") is not None:
            score += (4.0 - p["csat_mean"]) * 3
        score += (p.get("pct_escalated") or 0) / 5
        if p.get("median_resolution_hours") and med_res and not math.isnan(med_res):
            score += 2 * (p["median_resolution_hours"] / med_res - 1)
        return score

    ranked = sorted(payload, key=impact, reverse=True)
    n_high = max(1, len(ranked) // 3)
    priority = {p["cluster"]: ("high" if i < n_high else "medium" if i < 2 * n_high else "low") for i, p in enumerate(ranked)}

    clusters = []
    for p in payload:
        terms = [t.strip() for t in str(p.get("top_terms", "")).split(",") if t.strip()]
        actions = []
        if (p.get("trend_pp") or 0) > 2:
            actions.append(f"Volume share rose {p['trend_pp']} pp recently - check recent releases or incidents related to '{', '.join(terms[:3])}'.")
        if p.get("csat_mean") is not None and p["csat_mean"] < 3.5:
            actions.append(f"CSAT is low ({p['csat_mean']}) - review the macros and resolution steps agents use for this issue.")
        if p.get("median_resolution_hours") and med_res and p["median_resolution_hours"] > 1.5 * med_res:
            actions.append("Resolution is much slower than average - write a troubleshooting guide or route to a specialist queue.")
        if (p.get("pct_escalated") or 0) > 15:
            actions.append(f"{p['pct_escalated']}% of tickets escalate - add tier-1 diagnostics before escalation.")
        if not actions:
            actions.append("Publish a self-service help-center article and a reply macro for this topic.")
        summary_bits = [f"{p['n_tickets']} tickets ({p['share_pct']}% of volume)"]
        if p.get("median_resolution_hours") is not None:
            summary_bits.append(f"median resolution {p['median_resolution_hours']} h")
        if p.get("csat_mean") is not None:
            summary_bits.append(f"CSAT {p['csat_mean']}")
        clusters.append({
            "cluster": p["cluster"],
            "theme": " / ".join(terms[:3]).title() or f"Cluster {p['cluster']}",
            "summary": "; ".join(summary_bits) + ".",
            "likely_root_cause": "Not inferred in offline mode - run with the Claude API step enabled for a root-cause hypothesis.",
            "recommended_actions": actions,
            "priority": priority[p["cluster"]],
        })
    top = [f"Cluster {p['cluster']}: " + " / ".join(str(p.get("top_terms", "")).split(", ")[:3]) for p in ranked[:n_high]]
    return {
        "source": "offline-heuristic",
        "overall_summary": f"{len(payload)} ticket themes found. Highest-impact clusters ranked by volume, recent growth, CSAT, escalations and resolution time.",
        "top_priorities": top,
        "clusters": clusters,
    }


#  Entry point

def generate_insights(
    profiles: pd.DataFrame,
    samples: Dict[int, List[str]],
    context: dict,
    use_api: bool = True,
    model: str = DEFAULT_MODEL,
    log: Callable[[str], None] = print,
) -> dict:
    payload = build_cluster_payload(profiles, samples)
    note: Optional[str] = None
    report = None
    if use_api:
        try:
            report = _call_claude(payload, context, model)
        except ImportError:
            note = "The 'anthropic' package is not installed (pip install anthropic); used offline insights."
        except Exception as e:  # auth, network, rate limit, refusal: fall back so the demo keeps running
            note = f"Claude API call failed ({type(e).__name__}: {e}); used offline insights."
    else:
        note = "Cloud API step disabled (--no-api); used offline insights."
    if report is None:
        if note:
            log(f"  ! {note}")
        report = _offline_report(payload)

   
    by_cluster = {p["cluster"]: p for p in payload}
    for c in report["clusters"]:
        p = by_cluster.get(c["cluster"], {})
        c["metrics"] = {k: v for k, v in p.items() if k not in ("cluster", "sample_tickets", "top_terms")}
        c["top_terms"] = [t.strip() for t in str(p.get("top_terms", "")).split(",") if t.strip()]
        c["sample_tickets"] = p.get("sample_tickets", [])
    report["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["note"] = note
    return report
