"""Merge DHS cluster-level geospatial covariates onto the survey frame.

This is the guaranteed, no-Earth-Engine climate layer. It joins the DHS
Geospatial Covariates file (IAGC7AFL.csv) to the survey frame on cluster id
and derives a small set of climate-normal features.

What these features are, and are not
------------------------------------
They are long-run climate normals and 5-yearly snapshots. They describe the
*typical* thermal environment of a woman's survey cluster. They are NOT the
acute heat she experienced in the months before her interview -- that is what
the Earth Engine layer (climate_gee.py) adds. Keeping the two separate lets
Week 3 ask which representation predicts better, which is the interesting
question rather than a nuisance.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    DATA_RAW,
    GEOCOV_FEATURES,
    GEOCOV_FILE,
    GEOCOV_KEY,
    GEOCOV_MONTHLY_TEMP,
    SURVEY_DESIGN,
)

log = logging.getLogger(__name__)

CLUSTER = SURVEY_DESIGN["cluster"]  # v001


def load_covariates(path: str | Path | None = None) -> pd.DataFrame:
    """Read the DHS geospatial covariates CSV, keeping only needed columns."""
    path = Path(path) if path else DATA_RAW / GEOCOV_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place the DHS Geospatial Covariates file "
            "(IAGC7AFL.csv) in data/raw/."
        )

    header = pd.read_csv(path, nrows=0).columns.tolist()
    wanted = [GEOCOV_KEY] + GEOCOV_FEATURES + GEOCOV_MONTHLY_TEMP
    present = [c for c in wanted if c in header]
    missing = [c for c in wanted if c not in header]
    if GEOCOV_KEY not in header:
        raise KeyError(
            f"Merge key {GEOCOV_KEY!r} absent from {path.name}. "
            f"Columns look like: {header[:8]}"
        )
    if missing:
        warnings.warn(
            f"{len(missing)} covariate columns absent and will be skipped: "
            f"{missing}",
            stacklevel=2,
        )

    df = pd.read_csv(path, usecols=present)
    # DHS uses large negative sentinels for missing covariates
    df = df.replace({-9999: np.nan, -9998: np.nan})
    log.info("Loaded covariates %s: %d clusters x %d cols", path.name, len(df), df.shape[1])
    return df


def derive_climate_features(cov: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features from the raw covariate columns."""
    out = cov.copy()

    monthly = [c for c in GEOCOV_MONTHLY_TEMP if c in out.columns]
    if len(monthly) == 12:
        m = out[monthly]
        out["temp_annual_mean"] = m.mean(axis=1)
        out["temp_annual_range"] = m.max(axis=1) - m.min(axis=1)
        # Hot season = mean of the three hottest months for that cluster.
        # Row-wise, so it respects that peak heat falls in different months
        # across India.
        out["temp_hot3_mean"] = np.sort(m.to_numpy(), axis=1)[:, -3:].mean(axis=1)
        out["temp_hottest_month"] = m.max(axis=1)
    else:
        warnings.warn(
            f"Only {len(monthly)}/12 monthly temperature columns present; "
            "skipping seasonal features.",
            stacklevel=2,
        )

    return out


def merge_onto_survey(
    survey: pd.DataFrame, cov: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Left-join covariates onto the survey frame on cluster id.

    Left join keeps every woman; clusters with no covariate row get NaN climate
    features, which Week 3 handles explicitly rather than silently dropping.
    """
    if cov is None:
        cov = derive_climate_features(load_covariates())

    if CLUSTER not in survey.columns:
        raise KeyError(
            f"Survey frame has no cluster column {CLUSTER!r}. Rebuild the frame "
            "with the updated loader (config now requests v001)."
        )

    cov = cov.rename(columns={GEOCOV_KEY: CLUSTER})
    before = len(survey)
    merged = survey.merge(cov, on=CLUSTER, how="left", validate="many_to_one")

    matched = merged["temp_annual_mean"].notna().mean() if "temp_annual_mean" in merged else np.nan
    log.info(
        "Merged covariates: %d rows (unchanged=%s), climate non-missing=%.1f%%",
        len(merged), len(merged) == before, 100 * matched,
    )
    if not np.isnan(matched) and matched < 0.90:
        warnings.warn(
            f"Only {matched:.1%} of women matched a climate cluster. Check that "
            "survey v001 and covariate DHSCLUST use the same cluster numbering.",
            stacklevel=2,
        )
    return merged
