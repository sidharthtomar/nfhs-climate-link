#!/usr/bin/env python
"""Week 3.5: district menstrual-hygiene need map.

Produces the usable deliverable -- a ranked district table (raw need + adjusted
need) as CSV, plus a choropleth if the district shapefile is available.

Usage:
    python scripts/04_district_map.py
    python scripts/04_district_map.py --shapefile data/raw/districts.shp
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nfhs_climate.config import DATA_PROCESSED, FIGURES, TABLES  # noqa: E402
from nfhs_climate.models.features import assemble  # noqa: E402
from nfhs_climate.models.models import estimator_factories  # noqa: E402
from nfhs_climate.models.needmap import DISTRICT, build_need_map  # noqa: E402


def _oof_predictions(df, outcome):
    """Out-of-fold spatial predictions, so 'predicted' is never in-sample."""
    from sklearn.model_selection import GroupKFold

    X, y, groups, meta = assemble(df, outcome=outcome)
    fac = estimator_factories(meta["categorical"], meta["numeric"])["gradient_boosting"]
    pred = pd.Series(np.nan, index=X.index)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        est = fac()
        w = meta["survey_weight"]
        kw = {"clf__sample_weight": w.iloc[tr].to_numpy()} if w is not None else {}
        est.fit(X.iloc[tr], y.iloc[tr], **kw)
        pred.iloc[te] = est.predict_proba(X.iloc[te])[:, 1]
    out = df.loc[X.index].copy()
    out["predicted_prob"] = pred.values
    return out


def _plot_choropleth(need, shp_path, path):
    try:
        import geopandas as gpd
    except ImportError:
        print("geopandas not installed; skipping choropleth (CSV still written).")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gdf = gpd.read_file(shp_path)
    # join key differs by shapefile; try common district-code columns
    key = next((c for c in gdf.columns if c.lower() in
                ("dtcode11", "censuscode", "dist_code", "district_c")), None)
    if key is None:
        print("Could not find a district-code column in the shapefile; "
              "skipping map. CSV still written.")
        return
    merged = gdf.merge(need, left_on=key, right_on=DISTRICT, how="left")
    fig, ax = plt.subplots(figsize=(8, 9))
    merged.plot(
        column="hygienic_rate", cmap="RdYlGn", legend=True, ax=ax,
        missing_kwds={"color": "lightgrey"},
        legend_kwds={"label": "Hygienic-material use rate"},
    )
    ax.set_title("Menstrual hygiene practice by district (NFHS-5)\nred = highest need")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_PROCESSED / "analysis_full.parquet"))
    ap.add_argument("--outcome", default="exclusive_hygienic_narrow")
    ap.add_argument("--shapefile", default=None, help="Optional district shapefile for the map")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    data_path = Path(args.data)
    if not data_path.exists():
        alt = DATA_PROCESSED / "analysis_base.parquet"
        data_path = alt if alt.exists() else data_path
    if not data_path.exists():
        print("No analysis parquet found. Run 02a first.")
        return 1

    df = pd.read_parquet(data_path)

    print("Computing out-of-fold predictions for adjusted-need ranking...")
    scored = _oof_predictions(df, args.outcome)

    need = build_need_map(scored, outcome=args.outcome, predicted_col="predicted_prob")

    out_csv = TABLES / "district_need_map.csv"
    need.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  ({len(need)} districts)")

    print("\nTop 15 highest-need districts (lowest hygienic-material use):")
    cols = [DISTRICT, "state", "n_women", "hygienic_rate", "raw_need_rank"]
    if "underperforming" in need:
        cols += ["adjusted_gap", "underperforming"]
    print(need[cols].head(15).to_string(index=False))

    if args.shapefile:
        _plot_choropleth(need, args.shapefile, FIGURES / "district_need_map.png")
        print(f"Wrote choropleth to {FIGURES/'district_need_map.png'}")
    else:
        print("\nNo --shapefile given; CSV written without the map. Add a district "
              "boundary shapefile to render the choropleth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
