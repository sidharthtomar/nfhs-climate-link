#!/usr/bin/env python
"""Week 2a: merge DHS geospatial covariates onto the survey frame.

No Earth Engine. This alone produces a complete, trainable dataset -- the
guaranteed path that does not depend on GEE behaving.

Usage:
    python scripts/02a_merge_covariates.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfhs_climate.config import DATA_INTERIM, DATA_PROCESSED  # noqa: E402
from nfhs_climate.data.covariates import (  # noqa: E402
    derive_climate_features,
    load_covariates,
    merge_onto_survey,
)
import pandas as pd  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    survey_path = DATA_INTERIM / "survey_frame.parquet"
    if not survey_path.exists():
        print(f"Missing {survey_path}. Run scripts/01_build_survey.py first.")
        return 1

    survey = pd.read_parquet(survey_path)
    cov = derive_climate_features(load_covariates())
    merged = merge_onto_survey(survey, cov)

    out = DATA_PROCESSED / "analysis_base.parquet"
    merged.to_parquet(out, index=False)
    print(f"\nWrote {out}  ({len(merged):,} rows x {merged.shape[1]} cols)")
    print("This dataset is trainable now. The GEE heat layer (02b) is additive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
