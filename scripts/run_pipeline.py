"""
Runs the whole thing on a ticket file and drops the results in outputs.

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from support_analysis.config import PipelineConfig  # noqa: E402
from support_analysis.insights import DEFAULT_MODEL  # noqa: E402
from support_analysis.io import read_table  # noqa: E402
from support_analysis.pipeline import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Ticket export (.csv or .xlsx)")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256, help="Tickets per K-means training batch")
    ap.add_argument("--epochs", type=int, default=3, help="Passes over the batches")
    ap.add_argument("--no-api", action="store_true", help="Skip the Claude API call and use offline insights")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Claude model for the insight step")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = PipelineConfig(
        random_state=args.seed, k_min=args.k_min, k_max=args.k_max,
        batch_size=args.batch_size, n_epochs=args.epochs,
        use_api=not args.no_api, model=args.model,
    )
    run(read_table(args.input), cfg, outdir=args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()
