"""The K-means part. Picks how many groups to make, trains on tickets a batch at a
time, then sums up each group (size, trend, resolution time, CSAT, example tickets)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


@dataclass
class SegmentationResult:
    k: int
    model: MiniBatchKMeans
    labels: np.ndarray
    silhouette: float
    batch_log: List[dict] = field(default_factory=list)


def _silhouette(X: np.ndarray, labels: np.ndarray, sample_size: int, random_state: int) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(X, labels, sample_size=min(sample_size, len(X)), random_state=random_state))


def choose_k_by_silhouette(
    X: np.ndarray,
    k_min: int = 3,
    k_max: int = 10,
    random_state: int = 42,
    batch_size: int = 256,
    sample_size: int = 2000,
) -> Tuple[int, Dict[int, float]]:
    """Fit a quick model for each K and keep the one with the best (sampled) silhouette score."""
    k_max = min(k_max, len(X) - 1)
    if k_min < 2 or k_min > k_max:
        raise ValueError(f"Invalid K range: k_min={k_min}, k_max={k_max} (n_tickets={len(X)})")
    scores: Dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        km = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, n_init=3, random_state=random_state)
        scores[k] = _silhouette(X, km.fit_predict(X), sample_size, random_state)
    finite = {k: s for k, s in scores.items() if np.isfinite(s)}
    best_k = max(finite, key=finite.get) if finite else k_min
    return best_k, scores


def make_batches(n: int, batch_size: int, min_size: int) -> List[np.ndarray]:
    """Consecutive index batches; a too-small final batch is merged into the previous one."""
    batches = [np.arange(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]
    if len(batches) > 1 and len(batches[-1]) < min_size:
        batches[-2] = np.concatenate([batches[-2], batches.pop()])
    return batches


def fit_kmeans_in_batches(
    X: np.ndarray,
    k: int,
    batch_size: int = 256,
    n_epochs: int = 3,
    random_state: int = 42,
    sample_size: int = 2000,
) -> SegmentationResult:
    """Stream tickets (in arrival order) through MiniBatchKMeans.partial_fit.

    This is the same loop a production job would run as new batches of tickets
    arrive; the saved model can later `predict` clusters for unseen batches.
    """
    batch_size = max(batch_size, 3 * k)  # the first batch must be large enough to initialise k centres
    model = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=random_state)
    batches = make_batches(len(X), batch_size, min_size=k)
    log: List[dict] = []
    for epoch in range(1, n_epochs + 1):
        for b, idx in enumerate(batches, start=1):
            model.partial_fit(X[idx])
            log.append({
                "epoch": epoch,
                "batch": b,
                "n_tickets": int(len(idx)),
                "batch_inertia": round(float(-model.score(X[idx])), 4),
            })
    labels = model.predict(X)
    return SegmentationResult(
        k=k, model=model, labels=labels,
        silhouette=_silhouette(X, labels, sample_size, random_state), batch_log=log,
    )


def representative_samples(
    X: np.ndarray, labels: np.ndarray, centers: np.ndarray, texts: pd.Series, n: int = 5, max_chars: int = 400
) -> Dict[int, List[str]]:
    """The n tickets closest to each cluster centre (already PII-redacted)."""
    out: Dict[int, List[str]] = {}
    texts = texts.reset_index(drop=True)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        dist = np.linalg.norm(X[idx] - centers[c], axis=1)
        out[int(c)] = [texts.iloc[i][:max_chars] for i in idx[np.argsort(dist)[:n]]]
    return out


def _top_value(s: pd.Series) -> str:
    vc = s.dropna().value_counts()
    return f"{vc.index[0]} ({vc.iloc[0] / vc.sum():.0%})" if len(vc) else ""


def profile_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    top_terms: Dict[int, List[str]],
    created_col: str = "created_at",
    recent_quantile: float = 0.75,
) -> pd.DataFrame:
    """Support-oriented metrics per cluster: volume, recent trend, resolution time, CSAT, escalations."""
    d = df.reset_index(drop=True).copy()
    d["cluster"] = labels
    n_total = len(d)

    recent = None
    if created_col in d.columns and d[created_col].notna().any():
        cutoff = d[created_col].quantile(recent_quantile)
        recent = d[created_col] >= cutoff

    rows = []
    for c, g in d.groupby("cluster"):
        row = {
            "cluster": int(c),
            "n_tickets": int(len(g)),
            "share_pct": round(100 * len(g) / n_total, 1),
            "top_terms": ", ".join(top_terms.get(int(c), [])),
        }
        if recent is not None:
            r, e = recent.loc[g.index], ~recent
            recent_share = r.sum() / max(recent.sum(), 1)
            earlier_share = (e.loc[g.index]).sum() / max(e.sum(), 1)
            row["trend_pp"] = round(100 * (recent_share - earlier_share), 1)
        if "resolution_hours" in g:
            row["median_resolution_hours"] = round(float(g["resolution_hours"].median()), 1) if g["resolution_hours"].notna().any() else None
        if "csat" in g:
            row["csat_mean"] = round(float(g["csat"].mean()), 2) if g["csat"].notna().any() else None
            row["csat_response_rate_pct"] = round(100 * g["csat"].notna().mean(), 1)
        if "priority" in g:
            row["pct_high_or_urgent"] = round(100 * g["priority"].isin(["high", "urgent"]).mean(), 1)
        if "status" in g:
            row["pct_escalated"] = round(100 * (g["status"] == "escalated").mean(), 1)
            row["pct_open"] = round(100 * (g["status"] == "open").mean(), 1)
        for col in ("channel", "product"):
            if col in g:
                row[f"top_{col}"] = _top_value(g[col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_tickets", ascending=False).reset_index(drop=True)


def cluster_trend(df: pd.DataFrame, labels: np.ndarray, created_col: str = "created_at") -> pd.DataFrame:
    """Weekly ticket counts per cluster."""
    if created_col not in df.columns:
        return pd.DataFrame(columns=["week", "cluster", "n_tickets"])
    d = pd.DataFrame({"created": df[created_col].reset_index(drop=True), "cluster": labels}).dropna(subset=["created"])
    created = d["created"].dt.tz_convert(None) if d["created"].dt.tz is not None else d["created"]
    d["week"] = created.dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
    return d.groupby(["week", "cluster"]).size().rename("n_tickets").reset_index()
