"""Glues all the steps together: load -> clean -> vectorize -> K-means -> Claude -> save.
The command-line script and the Streamlit app both run through here."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .cleaning import TicketCleaningReport, clean_tickets
from .config import PipelineConfig
from .eda import run_basic_eda
from .features import TextVectorizer
from .insights import generate_insights
from .report import build_report
from .segmentation import (
    SegmentationResult,
    choose_k_by_silhouette,
    cluster_trend,
    fit_kmeans_in_batches,
    profile_clusters,
    representative_samples,
)
from .utils import ensure_dir, save_joblib, save_json


@dataclass
class PipelineResult:
    tickets: pd.DataFrame              # cleaned tickets + cluster label
    cleaning_report: TicketCleaningReport
    vectorizer: TextVectorizer
    k_scores: Dict[int, float]
    segmentation: SegmentationResult
    profiles: pd.DataFrame
    trend: pd.DataFrame
    samples: Dict[int, List[str]]
    insights: dict
    summary: dict = field(default_factory=dict)


def step_clean(raw: pd.DataFrame, cfg: PipelineConfig):
    return clean_tickets(
        raw, id_col=cfg.id_col, text_cols=cfg.text_cols, created_col=cfg.created_col,
        resolved_col=cfg.resolved_col, min_text_chars=cfg.min_text_chars,
    )


def step_cluster(tickets: pd.DataFrame, cfg: PipelineConfig):
    vec = TextVectorizer(max_features=cfg.max_features, svd_components=cfg.svd_components, random_state=cfg.random_state)
    X = vec.fit_transform(tickets["text"].tolist())
    best_k, scores = choose_k_by_silhouette(
        X, k_min=cfg.k_min, k_max=cfg.k_max, random_state=cfg.random_state,
        batch_size=cfg.batch_size, sample_size=cfg.silhouette_sample,
    )
    seg = fit_kmeans_in_batches(
        X, k=best_k, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
        random_state=cfg.random_state, sample_size=cfg.silhouette_sample,
    )
    centers = seg.model.cluster_centers_
    terms = dict(enumerate(vec.top_terms(centers)))
    profiles = profile_clusters(tickets, seg.labels, terms, created_col=cfg.created_col)
    trend = cluster_trend(tickets, seg.labels, created_col=cfg.created_col)
    samples = representative_samples(X, seg.labels, centers, tickets["text"], n=cfg.samples_per_cluster)
    return vec, X, scores, seg, profiles, trend, samples


def dataset_context(raw: pd.DataFrame, tickets: pd.DataFrame, k: int, silhouette: float, cfg: PipelineConfig) -> dict:
    """Dataset-level facts sent to the insight step alongside the cluster profiles."""
    ctx = {"n_tickets_raw": int(len(raw)), "n_tickets_clean": int(len(tickets)), "k": int(k),
           "silhouette": round(float(silhouette), 3)}
    if cfg.created_col in tickets and tickets[cfg.created_col].notna().any():
        ctx["date_from"] = str(tickets[cfg.created_col].min().date())
        ctx["date_to"] = str(tickets[cfg.created_col].max().date())
        ctx["trend_pp_definition"] = "share of tickets in the most recent 25% of the period minus share in the earlier 75%, in percentage points"
    if "resolution_hours" in tickets and tickets["resolution_hours"].notna().any():
        ctx["overall_median_resolution_hours"] = round(float(tickets["resolution_hours"].median()), 1)
    if "csat" in tickets and tickets["csat"].notna().any():
        ctx["overall_mean_csat"] = round(float(tickets["csat"].mean()), 2)
    return ctx


def run(raw: pd.DataFrame, cfg: PipelineConfig, outdir: Optional[str | Path] = None,
        log: Callable[[str], None] = print) -> PipelineResult:
    log(f"[1/5] Ingest     {len(raw):,} raw rows, {raw.shape[1]} columns")

    tickets, report = step_clean(raw, cfg)
    log(f"[2/5] Clean      {report.n_rows_out:,} tickets kept | "
        f"{report.basic.duplicates_removed} duplicate IDs, {report.content_duplicates_removed} duplicate content, "
        f"{report.empty_or_short_dropped} empty removed | PII redacted in {report.tickets_with_pii} tickets")

    vec, X, scores, seg, profiles, trend, samples = step_cluster(tickets, cfg)
    n_batches = max(e["batch"] for e in seg.batch_log)
    log(f"[3/5] Vectorize  TF-IDF -> {X.shape[1]}-dim LSA vectors")
    log(f"[4/5] K-means    K={seg.k} (silhouette {seg.silhouette:.3f}), trained on {n_batches} batches x {cfg.n_epochs} epochs")

    ctx = dataset_context(raw, tickets, seg.k, seg.silhouette, cfg)
    insights = generate_insights(profiles, samples, ctx, use_api=cfg.use_api, model=cfg.model, log=log)
    log(f"[5/5] Insights   {len(insights['clusters'])} cluster insights from {insights['source']}")

    tickets = tickets.copy()
    tickets["cluster"] = seg.labels
    summary = {
        **ctx,
        "batch_size": cfg.batch_size, "n_batches": n_batches, "n_epochs": cfg.n_epochs,
        "vector_dims": int(X.shape[1]), "insight_source": insights["source"],
        "generated_at": insights["generated_at"],
    }
    result = PipelineResult(tickets, report, vec, scores, seg, profiles, trend, samples, insights, summary)
    if outdir is not None:
        paths = write_artifacts(result, outdir)
        log("Artifacts:")
        for p in paths:
            log(f"  - {p}")
    return result


def write_artifacts(res: PipelineResult, outdir: str | Path) -> List[Path]:
    outdir = Path(outdir)
    data_dir, models_dir, reports_dir = (ensure_dir(outdir / d) for d in ("data", "models", "reports"))
    paths = {
        "tickets": data_dir / "tickets_clustered.csv",
        "cleaning": reports_dir / "cleaning_report.json",
        "k_scores": reports_dir / "k_silhouette_scores.json",
        "batches": reports_dir / "batch_log.json",
        "profiles": reports_dir / "cluster_profiles.csv",
        "trend": reports_dir / "cluster_trend.csv",
        "report": reports_dir / "insights_report.md",
        "insights": reports_dir / "insights.json",
        "summary": reports_dir / "run_summary.json",
        "vectorizer": models_dir / "text_vectorizer.joblib",
        "kmeans": models_dir / "kmeans.joblib",
    }
    res.tickets.to_csv(paths["tickets"], index=False)
    save_json(res.cleaning_report.to_dict(), paths["cleaning"])
    save_json({str(k): (None if not np.isfinite(v) else round(float(v), 4)) for k, v in res.k_scores.items()}, paths["k_scores"])
    save_json(res.segmentation.batch_log, paths["batches"])
    res.profiles.to_csv(paths["profiles"], index=False)
    res.trend.to_csv(paths["trend"], index=False)
    save_json(res.insights, paths["insights"])
    save_json(res.summary, paths["summary"])
    paths["report"].write_text(build_report(res.insights, res.summary, res.cleaning_report.to_dict()), encoding="utf-8")
    save_joblib(res.vectorizer, paths["vectorizer"])
    save_joblib(res.segmentation.model, paths["kmeans"])
    run_basic_eda(res.tickets, reports_dir / "eda")
    return list(paths.values())


def assign_new_batch(raw: pd.DataFrame, vectorizer: TextVectorizer, kmeans, cfg: PipelineConfig) -> pd.DataFrame:
    """Clean a new batch of tickets and assign each one to an existing cluster (no retraining)."""
    tickets, _ = step_clean(raw, cfg)
    tickets["cluster"] = kmeans.predict(vectorizer.transform(tickets["text"].tolist()))
    return tickets
