"""
Quick does it all still work  check. Runs everything on 600 fake tickets, no API call.

"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    tmp = Path(tempfile.mkdtemp())
    raw = tmp / "tickets.csv"
    outdir = tmp / "outputs"

    subprocess.check_call([sys.executable, str(ROOT / "scripts/generate_sample_data.py"), "--out", str(raw), "--n", "600"])
    subprocess.check_call([sys.executable, str(ROOT / "scripts/run_pipeline.py"), "--input", str(raw),
                           "--outdir", str(outdir), "--no-api", "--batch-size", "128"])
    subprocess.check_call([sys.executable, str(ROOT / "scripts/score_new_batch.py"), "--input", str(raw), "--outdir", str(outdir)])

    for rel in ["data/tickets_clustered.csv", "reports/cluster_profiles.csv", "reports/insights.json",
                "reports/run_summary.json", "models/kmeans.joblib", "models/text_vectorizer.joblib"]:
        assert (outdir / rel).exists(), f"missing {rel}"
    insights = json.loads((outdir / "reports/insights.json").read_text(encoding="utf-8"))
    assert insights["clusters"], "no cluster insights"
    cleaned = (outdir / "data/tickets_clustered.csv").read_text(encoding="utf-8")
    assert "@example.com" not in cleaned, "PII leaked into cleaned output"
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
