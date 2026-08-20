#!/usr/bin/env python
"""Week 1: build the survey frame and validate it against published figures.

Usage:
    python scripts/01_build_survey.py --recode data/raw/IAIR7EFL.DTA

Nothing downstream should run until this exits cleanly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfhs_climate.config import DATA_INTERIM, TABLES  # noqa: E402
from nfhs_climate.data.nfhs import build_analysis_frame  # noqa: E402
from nfhs_climate.data.validate import prevalence_table, report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recode", required=True, help="Path to DHS individual recode")
    ap.add_argument(
        "--no-strict",
        action="store_true",
        help="Warn instead of raising when benchmarks fail (diagnostics only)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    df = build_analysis_frame(args.recode)

    checks = report(df, strict=not args.no_strict)
    checks.to_csv(TABLES / "benchmark_checks.csv", index=False)

    prev = prevalence_table(df)
    prev.to_csv(TABLES / "outcome_definitions.csv", index=False)
    print("\nOutcome definition sensitivity:")
    print(prev.to_string(index=False))

    out = DATA_INTERIM / "survey_frame.parquet"
    df.to_parquet(out, index=False)
    print(f"\nWrote {out}  ({len(df):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
