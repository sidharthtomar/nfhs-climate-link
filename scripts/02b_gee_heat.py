#!/usr/bin/env python
"""Week 2b: interview-windowed extreme-heat exposure via Earth Engine.

Additive. Reads cluster GPS coordinates from the DHS GPS shapefile, attaches
each cluster's interview month/year from the survey frame, computes heat-day
counts on Google's servers, and merges the result into the analysis base.

Run 02a first. This step needs `earthengine authenticate` to have succeeded.

Usage:
    python scripts/02b_gee_heat.py --gps data/raw/IAGE7AFL.shp
    python scripts/02b_gee_heat.py --gps data/raw/IAGE7AFL.shp --refresh   # recompute
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from nfhs_climate.config import DATA_PROCESSED, SURVEY_DESIGN  # noqa: E402
from nfhs_climate.data.climate_gee import get_cluster_exposure  # noqa: E402

CLUSTER = SURVEY_DESIGN["cluster"]
IV_YEAR = SURVEY_DESIGN["interview_year"]    # v007
IV_MONTH = SURVEY_DESIGN["interview_month"]  # v006


def load_gps(path: Path) -> pd.DataFrame:
    """Read cluster centroids from the DHS GPS shapefile."""
    import geopandas as gpd

    gdf = gpd.read_file(path)
    # DHS GPS files carry DHSCLUST, LATNUM, LONGNUM. Drop clusters with no fix
    # (coded 0,0) -- a known DHS convention for missing coordinates.
    gdf = gdf[(gdf["LATNUM"] != 0) | (gdf["LONGNUM"] != 0)]
    return pd.DataFrame(gdf[["DHSCLUST", "LATNUM", "LONGNUM"]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gps", required=True, help="Path to DHS GPS shapefile (.shp)")
    ap.add_argument("--refresh", action="store_true", help="Recompute, ignore cache")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    base_path = DATA_PROCESSED / "analysis_base.parquet"
    if not base_path.exists():
        print(f"Missing {base_path}. Run scripts/02a_merge_covariates.py first.")
        return 1

    base = pd.read_parquet(base_path)

    # One interview date per cluster (mode within cluster)
    dates = (
        base.groupby(CLUSTER)[[IV_YEAR, IV_MONTH]]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
        .rename(columns={IV_YEAR: "interview_year", IV_MONTH: "interview_month"})
    )

    gps = load_gps(Path(args.gps))
    clusters = gps.merge(
        dates, left_on="DHSCLUST", right_on=CLUSTER, how="inner"
    )
    print(f"Computing heat exposure for {len(clusters):,} clusters via Earth Engine...")
    print("This runs server-side and may take several minutes. Cached afterwards.")

    exposure = get_cluster_exposure(clusters, refresh=args.refresh)

    merged = base.merge(
        exposure, left_on=CLUSTER, right_on="DHSCLUST", how="left"
    ).drop(columns=["DHSCLUST"], errors="ignore")

    matched = merged["heat_days_12mo"].notna().mean()
    print(f"Heat exposure attached to {matched:.1%} of women.")

    out = DATA_PROCESSED / "analysis_full.parquet"
    merged.to_parquet(out, index=False)
    print(f"\nWrote {out}  ({len(merged):,} rows x {merged.shape[1]} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
