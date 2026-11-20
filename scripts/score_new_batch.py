"""
New tickets --> This sorts them into the groups from the last run, without retraining.

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from support_analysis.config import PipelineConfig  # noqa: E402
from support_analysis.io import read_table, write_table  # noqa: E402
from support_analysis.pipeline import assign_new_batch  # noqa: E402
from support_analysis.utils import load_joblib, load_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="New ticket batch (.csv or .xlsx)")
    ap.add_argument("--outdir", default="outputs", help="Directory of the previous pipeline run")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    vec = load_joblib(outdir / "models" / "text_vectorizer.joblib")
    km = load_joblib(outdir / "models" / "kmeans.joblib")
    themes = {c["cluster"]: c["theme"] for c in load_json(outdir / "reports" / "insights.json")["clusters"]}

    tickets = assign_new_batch(read_table(args.input), vec, km, PipelineConfig())
    tickets["theme"] = tickets["cluster"].map(themes)
    dest = outdir / "data" / f"{Path(args.input).stem}_scored.csv"
    write_table(tickets, dest)

    print(f"Assigned {len(tickets)} tickets -> {dest}")
    print(tickets.groupby(["cluster", "theme"]).size().rename("n_tickets").sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
