"""Load NFHS-5 individual recode and construct the analysis frame.

Design principle: fail loudly and early. Survey recodes are full of variables
that look right and are not, so every step that could silently mis-code the
outcome raises instead of guessing.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    AGE_MAX,
    AGE_MIN,
    COVARIATES,
    HYGIENIC_BROAD,
    HYGIENIC_NARROW,
    NEGATIVE_CONTROL,
    PROTECTION_ITEMS,
    SURVEY_DESIGN,
)

log = logging.getLogger(__name__)


def load_recode(path: str | Path, round_name: str = "nfhs5") -> pd.DataFrame:
    """Read a DHS individual recode (.DTA or .SAV) keeping only needed columns.

    The full women's file is ~700k rows x ~5000 columns. Reading all of it
    will exhaust memory on a laptop, so we select columns up front.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place the DHS individual recode in data/raw/. "
            "Expected something like IAIR7EFL.DTA"
        )

    wanted = (
        list(PROTECTION_ITEMS[round_name])
        + list(SURVEY_DESIGN.values())
        + list(COVARIATES.values())
        + [NEGATIVE_CONTROL]
    )

    try:
        import pyreadstat
    except ImportError as exc:
        raise ImportError("pip install pyreadstat") from exc

    reader = {
        ".dta": pyreadstat.read_dta,
        ".sav": pyreadstat.read_sav,
    }.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Probe the header first so we can report missing variables by name
    _, meta = reader(path, metadataonly=True)
    available = set(meta.column_names)
    missing = [c for c in wanted if c not in available]
    if missing:
        warnings.warn(
            f"{len(missing)} configured variables absent from {path.name}: "
            f"{missing}. Check the recode manual and update config.py.",
            stacklevel=2,
        )
    usecols = [c for c in wanted if c in available]

    df, _ = reader(path, usecols=usecols)
    log.info("Loaded %s: %d rows x %d cols", path.name, len(df), df.shape[1])
    return df


def restrict_to_young_women(df: pd.DataFrame) -> pd.DataFrame:
    """Keep women aged 15-24 -- the only ones asked the menstrual items."""
    age = COVARIATES["age"]
    out = df[df[age].between(AGE_MIN, AGE_MAX)].copy()
    log.info("Age restriction %d-%d: %d -> %d rows", AGE_MIN, AGE_MAX, len(df), len(out))
    return out


def build_protection_flags(df: pd.DataFrame, round_name: str = "nfhs5") -> pd.DataFrame:
    """Rename raw protection variables to canonical labels.

    This function exists solely so that no downstream code ever touches a
    letter suffix. See the extended note in config.py -- a positional map
    across rounds sends 'nothing' onto 'menstrual cup'.
    """
    item_map = PROTECTION_ITEMS[round_name]
    present = {raw: label for raw, label in item_map.items() if raw in df.columns}

    if not present:
        raise KeyError(
            f"None of the expected protection variables {list(item_map)} are "
            "present. Wrong file, or the variable names differ in your extract."
        )
    if len(present) < len(item_map):
        warnings.warn(
            f"Only {len(present)}/{len(item_map)} protection items found: "
            f"missing {set(item_map) - set(present)}",
            stacklevel=2,
        )

    out = df.copy()
    for raw, label in present.items():
        # DHS multiple-response items are 0/1; anything else is missing
        out[f"uses_{label}"] = (
            pd.to_numeric(out[raw], errors="coerce").eq(1).astype("float")
        )
        out.loc[pd.to_numeric(out[raw], errors="coerce").isna(), f"uses_{label}"] = np.nan

    out.attrs["protection_labels"] = sorted(present.values())
    return out


def build_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the four outcome definitions used in the literature.

    Two axes, so four combinations:
      - narrow vs broad  : are locally prepared napkins 'hygienic'?
      - any vs exclusive : does using cloth as well disqualify you?

    Published prevalence ranges from ~50% to ~78% across these. Reporting
    only one and calling it 'menstrual hygiene' hides a 28-point choice.
    """
    labels = df.attrs.get("protection_labels")
    if labels is None:
        raise RuntimeError("Call build_protection_flags() before build_outcomes().")

    def cols_for(materials: set[str]) -> list[str]:
        return [f"uses_{m}" for m in materials if m in labels]

    all_cols = [f"uses_{m}" for m in labels]
    out = df.copy()

    # A respondent with no positive response on any item tells us nothing
    reported_any = out[all_cols].sum(axis=1, min_count=1) > 0

    for name, materials in (("narrow", HYGIENIC_NARROW), ("broad", HYGIENIC_BROAD)):
        hyg_cols = cols_for(materials)
        unhyg_cols = [c for c in all_cols if c not in hyg_cols]

        uses_hygienic = out[hyg_cols].sum(axis=1, min_count=1) > 0
        uses_unhygienic = out[unhyg_cols].sum(axis=1, min_count=1) > 0

        out[f"any_hygienic_{name}"] = np.where(
            reported_any, uses_hygienic.astype(float), np.nan
        )
        out[f"exclusive_hygienic_{name}"] = np.where(
            reported_any, (uses_hygienic & ~uses_unhygienic).astype(float), np.nan
        )

    out["survey_weight"] = pd.to_numeric(
        out[SURVEY_DESIGN["weight"]], errors="coerce"
    ) / 1e6

    return out


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    """Survey-weighted mean over non-missing rows."""
    mask = series.notna() & weights.notna()
    if not mask.any():
        return float("nan")
    return float(np.average(series[mask], weights=weights[mask]))


def build_analysis_frame(path: str | Path) -> pd.DataFrame:
    """End-to-end: raw recode -> analysis-ready frame for women aged 15-24."""
    df = load_recode(path)
    df = restrict_to_young_women(df)
    df = build_protection_flags(df)
    df = build_outcomes(df)
    return df
