"""
BUild readable report (insights_report.md) 
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from support_analysis.report import build_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="outputs", help="Directory of the pipeline run")
    args = ap.parse_args()

    reports = Path(args.outdir) / "reports"
    load = lambda name: json.loads((reports / name).read_text(encoding="utf-8")) if (reports / name).exists() else None  # noqa: E731
    insights = load("insights.json")
    if insights is None:
        sys.exit(f"No insights.json in {reports}. Run scripts/run_pipeline.py first.")

    dest = reports / "insights_report.md"
    dest.write_text(build_report(insights, load("run_summary.json"), load("cleaning_report.json")), encoding="utf-8")
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
