from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from support_analysis.io import write_table  # noqa: E402
from support_analysis.sample_data import make_sample  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="Output path (.csv or .xlsx)")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = make_sample(n=args.n, random_state=args.seed)
    write_table(df, args.out)
    print(f"Wrote sample tickets: {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
