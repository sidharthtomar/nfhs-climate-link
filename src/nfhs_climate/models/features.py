"""Assemble the modelling matrix from the analysis frame.

Turns the wide survey+climate frame into (X, y, groups) for modelling:
- y      : the primary outcome, exclusive_hygienic_narrow
- X      : socioeconomic + climate features, encoded
- groups : district id, for spatial cross-validation

Design choices worth knowing
----------------------------
- The outcome is narrow-exclusive (pads/tampons/cup only; cloth AND locally
  prepared napkins both disqualify). Chosen as primary because it is the
  strictest, most defensible "hygienic" definition. Broad is available for a
  sensitivity run.
- Climate features come in two families that the modelling deliberately keeps
  separable, so Week 3 can ask which matters:
    * normals   -- DHS geospatial covariates (typical climate of the district)
    * acute     -- GEE interview-windowed heat days (heat before interview)
- District id is carried through as the grouping variable, never as a feature.
  Putting district in X would let the model memorise district means, which is
  exactly the leakage spatial CV exists to expose.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import COVARIATES, GEOCOV_FEATURES

log = logging.getLogger(__name__)

PRIMARY_OUTCOME = "exclusive_hygienic_narrow"
BROAD_OUTCOME = "exclusive_hygienic_broad"

# Socioeconomic / individual predictors (DHS variable names as they appear in
# the frame). Deliberately EXCLUDES district and state -- those are grouping /
# geography, not individual features.
SES_FEATURES = [
    COVARIATES["age"],
    COVARIATES["education"],
    COVARIATES["wealth_quintile"],
    COVARIATES["residence"],
    COVARIATES["religion"],
    COVARIATES["caste"],
    COVARIATES["marital_status"],
    COVARIATES["age_at_menarche"],
    COVARIATES["media_newspaper"],
    COVARIATES["media_radio"],
    COVARIATES["media_tv"],
    COVARIATES["water_source"],
    COVARIATES["toilet_type"],
    COVARIATES["chw_menstrual_talk"],
]

# Climate-normal features (from DHS covariates + engineered in covariates.py)
CLIMATE_NORMAL_FEATURES = GEOCOV_FEATURES + [
    "temp_annual_mean",
    "temp_annual_range",
    "temp_hot3_mean",
    "temp_hottest_month",
]

# Acute heat features (from GEE). Absent if 02b was not run -- handled.
CLIMATE_ACUTE_FEATURES = ["heat_days_12mo", "heat_threshold_c"]

# Treated as categorical (one-hot). The rest are numeric.
CATEGORICAL = {
    COVARIATES["education"],
    COVARIATES["wealth_quintile"],
    COVARIATES["residence"],
    COVARIATES["religion"],
    COVARIATES["caste"],
    COVARIATES["marital_status"],
    COVARIATES["water_source"],
    COVARIATES["toilet_type"],
}

DISTRICT = COVARIATES["district"]


def assemble(
    df: pd.DataFrame,
    outcome: str = PRIMARY_OUTCOME,
    include_acute: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """Return (X, y, groups, meta).

    Rows with a missing outcome are dropped (can't train on them). Missing
    features are median/mode-imputed inside the model pipelines, not here, so
    the raw feature columns keep their NaNs at this stage.
    """
    if outcome not in df.columns:
        raise KeyError(f"Outcome {outcome!r} not in frame. Available outcomes: "
                       f"{[c for c in df.columns if 'hygienic' in c]}")

    frame = df[df[outcome].notna()].copy()

    feats = list(SES_FEATURES) + list(CLIMATE_NORMAL_FEATURES)
    have_acute = include_acute and all(c in frame.columns for c in CLIMATE_ACUTE_FEATURES)
    if have_acute:
        feats += CLIMATE_ACUTE_FEATURES
    else:
        log.warning(
            "Acute heat features (%s) not present -- running on climate normals "
            "only. Run scripts/02b_gee_heat.py to add them.",
            CLIMATE_ACUTE_FEATURES,
        )

    feats = [c for c in feats if c in frame.columns]
    missing = set(SES_FEATURES) - set(feats)
    if missing:
        log.warning("SES features missing from frame and skipped: %s", missing)

    X = frame[feats].copy()
    y = frame[outcome].astype(int)
    groups = frame[DISTRICT] if DISTRICT in frame.columns else pd.Series(
        np.arange(len(frame)), index=frame.index, name="row"
    )

    meta = {
        "n": len(frame),
        "outcome": outcome,
        "prevalence": float(y.mean()),
        "n_districts": int(groups.nunique()),
        "categorical": sorted(c for c in CATEGORICAL if c in feats),
        "numeric": sorted(c for c in feats if c not in CATEGORICAL),
        "climate_normal": [c for c in CLIMATE_NORMAL_FEATURES if c in feats],
        "climate_acute": [c for c in CLIMATE_ACUTE_FEATURES if c in feats],
        "has_acute": have_acute,
        "survey_weight": frame["survey_weight"] if "survey_weight" in frame else None,
    }
    log.info(
        "Assembled: n=%d, prevalence=%.3f, districts=%d, features=%d (acute=%s)",
        meta["n"], meta["prevalence"], meta["n_districts"], len(feats), have_acute,
    )
    return X, y, groups, meta
