"""Quick look at the cleaned data: summary stats, missing values, and a few histograms."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ID_LIKE = {"id", "ticket_id", "customer_id", "cluster"}


@dataclass
class EDAPaths:
    summary_path: Path
    missing_path: Path
    hist_dir: Path


def run_basic_eda(df: pd.DataFrame, outdir: str | Path) -> EDAPaths:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = outdir / "summary.csv"
    df.drop(columns=["text"], errors="ignore").describe(include="all").transpose().to_csv(summary_path)

    missing_path = outdir / "missing_rate.csv"
    df.isna().mean().sort_values(ascending=False).to_frame("missing_rate").to_csv(missing_path)

    hist_dir = outdir / "hist"
    hist_dir.mkdir(parents=True, exist_ok=True)
    for c in df.select_dtypes(include=["number"]).columns:
        if c in ID_LIKE:
            continue
        fig = plt.figure()
        plt.hist(df[c].dropna().values, bins=30)
        plt.title(f"Histogram: {c}")
        plt.xlabel(c)
        plt.ylabel("count")
        fig.savefig(hist_dir / f"{c}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    return EDAPaths(summary_path=summary_path, missing_path=missing_path, hist_dir=hist_dir)
