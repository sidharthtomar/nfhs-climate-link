"""District-level menstrual-hygiene need map.

The deliverable an NGO or health department could actually use: a ranked table
of districts by need, with climate and water/sanitation context, plus a flag
for districts doing WORSE than their socioeconomic profile predicts.

Two rankings, as decided:
- raw need      : weighted % using hygienic materials, lowest first. The direct
                  operational list -- "these districts are worst off".
- adjusted need : observed minus model-predicted rate. Negative = worse than a
                  district of this wealth/education profile "should" be. Finds
                  places where something beyond poverty is holding them back --
                  a targeting signal poverty maps miss.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import COVARIATES

log = logging.getLogger(__name__)

DISTRICT = COVARIATES["district"]
STATE = COVARIATES["state"]


def _wmean(v: pd.Series, w: pd.Series) -> float:
    m = v.notna() & w.notna()
    return float(np.average(v[m], weights=w[m])) if m.any() else np.nan


def build_need_map(
    df: pd.DataFrame,
    outcome: str = "exclusive_hygienic_narrow",
    predicted_col: str | None = None,
) -> pd.DataFrame:
    """One row per district: need, context, and (if available) adjusted need.

    df must contain the outcome, survey_weight, district id, and any context
    columns. predicted_col, if given, is a per-woman model probability used to
    compute the adjusted (residual) ranking.
    """
    if DISTRICT not in df.columns:
        raise KeyError(f"District column {DISTRICT!r} not in frame.")

    w = df["survey_weight"]
    ctx_cols = [
        c
        for c in ["temp_hot3_mean", "heat_days_12mo", "Aridity_2015",
                  COVARIATES["water_source"], COVARIATES["toilet_type"],
                  COVARIATES["wealth_quintile"]]
        if c in df.columns
    ]

    rows = []
    for did, g in df.groupby(DISTRICT):
        gw = g["survey_weight"]
        row = {
            DISTRICT: did,
            "state": g[STATE].iloc[0] if STATE in g else np.nan,
            "n_women": int(len(g)),
            "hygienic_rate": _wmean(g[outcome], gw),
        }
        for c in ctx_cols:
            row[f"ctx_{c}"] = _wmean(pd.to_numeric(g[c], errors="coerce"), gw)
        if predicted_col and predicted_col in g:
            row["predicted_rate"] = _wmean(g[predicted_col], gw)
        rows.append(row)

    out = pd.DataFrame(rows)

    # Raw need rank: lowest hygienic rate = highest need = rank 1
    out["raw_need_rank"] = out["hygienic_rate"].rank(method="min").astype("Int64")

    if "predicted_rate" in out.columns:
        # Adjusted need: observed - predicted. Negative = underperforming.
        out["adjusted_gap"] = out["hygienic_rate"] - out["predicted_rate"]
        out["adjusted_need_rank"] = out["adjusted_gap"].rank(method="min").astype("Int64")
        out["underperforming"] = out["adjusted_gap"] < 0

    out = out.sort_values("hygienic_rate").reset_index(drop=True)
    log.info(
        "Need map: %d districts, hygienic rate %.1f%%-%.1f%% (median %.1f%%)",
        len(out),
        100 * out["hygienic_rate"].min(),
        100 * out["hygienic_rate"].max(),
        100 * out["hygienic_rate"].median(),
    )
    return out
