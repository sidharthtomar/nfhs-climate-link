"""Interview-windowed extreme-heat exposure from ERA5-Land via Earth Engine.

This is the additive layer. It computes, for each survey cluster, the number of
extreme-heat days in the 12 months preceding the cluster's interview date,
where "extreme" is defined relative to that location's own 1991-2020
climatology. This is the acute-exposure measure the DHS climate normals cannot
provide.

Quota discipline
----------------
Everything heavy runs on Google's servers. We reduce ERA5-Land to a per-cluster
daily series server-side and export only a small table (one row per cluster).
We never pull rasters to the client. This keeps usage to a small fraction of
the free Community tier (150 EECU-hours).

Because Earth Engine work is slow and rate-limited, this module is written to
be run once and cached to disk. Week 3 reads the cached parquet, not GEE.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import (
    CLIMATOLOGY_YEARS,
    DATA_INTERIM,
    ERA5_LAND,
    GEE_PROJECT,
    HEAT_PERCENTILE,
)

log = logging.getLogger(__name__)

CACHE = DATA_INTERIM / "cluster_heat_exposure.parquet"


def init_ee() -> None:
    """Initialise Earth Engine. Assumes `earthengine authenticate` has run."""
    import ee

    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        # Surface the auth hint rather than a raw stack trace
        raise RuntimeError(
            "Earth Engine init failed. Run `earthengine authenticate` first. "
            "If you see a certificate error on macOS python.org builds, set "
            "SSL_CERT_FILE to certifi's bundle for the session."
        )
    log.info("Earth Engine initialised on project %s", GEE_PROJECT)


def _heat_index_celsius(img):
    """Approximate heat index from ERA5-Land 2m temp and dewpoint.

    ERA5-Land gives temperature_2m and dewpoint_2m in Kelvin. We convert to
    Celsius and relative humidity, then apply the Rothfusz regression used by
    the US NWS. A full WBGT would need radiation and wind; heat index is the
    standard choice in the NFHS heat literature and is what we validate against.
    """
    import ee

    t = img.select("temperature_2m").subtract(273.15)
    td = img.select("dewpoint_temperature_2m").subtract(273.15)

    # RH from temperature and dewpoint (Magnus formula)
    def sat(vp):
        return vp.multiply(17.625).divide(vp.add(243.04)).exp()

    rh = sat(td).divide(sat(t)).multiply(100).clamp(0, 100)

    tf = t.multiply(9 / 5).add(32)  # heat index regression is in Fahrenheit
    hi_f = (
        tf.multiply(-42.379 + 0)
        .add(tf.multiply(2.04901523))
        .add(rh.multiply(10.14333127))
        .add(tf.multiply(rh).multiply(-0.22475541))
        .add(tf.pow(2).multiply(-0.00683783))
        .add(rh.pow(2).multiply(-0.05481717))
        .add(tf.pow(2).multiply(rh).multiply(0.00122874))
        .add(tf.multiply(rh.pow(2)).multiply(0.00085282))
        .add(tf.pow(2).multiply(rh.pow(2)).multiply(-0.00000199))
        .add(-42.379)
    )
    hi_c = hi_f.subtract(32).multiply(5 / 9)
    # Below ~27C the regression is not meaningful; fall back to air temp
    return hi_c.max(t).rename("heat_index").copyProperties(img, ["system:time_start"])


def compute_cluster_exposure(
    clusters: pd.DataFrame,
    lat_col: str = "LATNUM",
    lon_col: str = "LONGNUM",
    id_col: str = "DHSCLUST",
    interview_year_col: str = "interview_year",
    interview_month_col: str = "interview_month",
    batch_size: int = 500,
) -> pd.DataFrame:
    """Server-side heat-day count per cluster in the year before interview.

    clusters must carry cluster id, GPS coords (from the GPS shapefile), and the
    interview month/year (from the survey frame). One row per cluster.

    Processed in batches because a single map over ~30k clusters, each filtering
    a 30-year daily collection and running a windowed reduction, overloads Earth
    Engine's expression graph. Batching keeps each request small and lets a
    transient failure retry just one chunk. Clusters whose climatology threshold
    samples as null (points over water/coastline with no ERA5-Land pixel) are
    dropped before the windowed count so nulls never reach Image.constant.
    """
    import ee

    init_ee()

    era5 = ee.ImageCollection(ERA5_LAND)
    y0, y1 = CLIMATOLOGY_YEARS

    # Precompute the per-location climatology threshold image ONCE, reused across
    # every batch. This is the expensive object; building it per batch would
    # waste quota.
    threshold_img = (
        era5.filterDate(f"{y0}-01-01", f"{y1}-12-31")
        .map(_heat_index_celsius)
        .reduce(ee.Reducer.percentile([HEAT_PERCENTILE]))
        .select([f"heat_index_p{HEAT_PERCENTILE}"], ["thr"])
    )

    def process_batch(chunk: pd.DataFrame) -> list[dict]:
        fc = ee.FeatureCollection(
            [
                ee.Feature(
                    ee.Geometry.Point([float(r[lon_col]), float(r[lat_col])]),
                    {
                        "cid": int(r[id_col]),
                        "yr": int(r[interview_year_col]),
                        "mo": int(r[interview_month_col]),
                    },
                )
                for _, r in chunk.iterrows()
            ]
        )

        # Sample the threshold at each point, then DROP nulls server-side so the
        # windowed count below never sees a null threshold.
        with_thr = threshold_img.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.first().setOutputs(["thr"]),
            scale=11132,
        ).filter(ee.Filter.notNull(["thr"]))

        def count_hot_days(feat):
            thr = ee.Number(feat.get("thr"))
            end = ee.Date.fromYMD(feat.get("yr"), feat.get("mo"), 1)
            start = end.advance(-1, "year")
            window = (
                era5.filterDate(start, end)
                .map(_heat_index_celsius)
                .map(lambda im: im.gte(thr).rename("hot"))
            )
            hot_days = window.sum().reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=feat.geometry(),
                scale=11132,
            ).get("hot")
            return feat.set({"heat_days": hot_days, "thr_c": thr})

        result = with_thr.map(count_hot_days)
        info = result.select(
            ["cid", "heat_days", "thr_c"], retainGeometry=False
        ).getInfo()
        return [f["properties"] for f in info["features"]]

    all_rows: list[dict] = []
    n = len(clusters)
    for i in range(0, n, batch_size):
        chunk = clusters.iloc[i : i + batch_size]
        try:
            rows = process_batch(chunk)
            all_rows.extend(rows)
            log.info(
                "  batch %d-%d/%d ok (%d returned, cumulative %d)",
                i, min(i + batch_size, n), n, len(rows), len(all_rows),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("  batch %d-%d FAILED: %s -- skipping", i, i + batch_size, exc)

    if not all_rows:
        raise RuntimeError(
            "No batches returned data. Check Earth Engine auth and that the GPS "
            "coordinates are valid lon/lat."
        )

    out = pd.DataFrame(all_rows).rename(
        columns={
            "cid": id_col,
            "heat_days": "heat_days_12mo",
            "thr_c": "heat_threshold_c",
        }
    )
    log.info("Computed heat exposure for %d of %d clusters", len(out), n)
    return out


def get_cluster_exposure(clusters: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Cached wrapper: compute once, then read parquet on subsequent runs."""
    if CACHE.exists() and not refresh:
        log.info("Loading cached heat exposure from %s", CACHE)
        return pd.read_parquet(CACHE)
    out = compute_cluster_exposure(clusters)
    out.to_parquet(CACHE, index=False)
    log.info("Cached heat exposure to %s", CACHE)
    return out
