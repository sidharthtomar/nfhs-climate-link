#!/usr/bin/env python
"""Week 3: modelling with honest (spatial) evaluation.

Trains a logistic baseline and a gradient-boosting model, evaluates each under
BOTH random k-fold and district-grouped (spatial) CV, and reports the leakage
gap. Also runs an ablation -- SES only vs SES+climate normals vs +acute heat --
so the marginal value of the climate features is explicit. Writes tables and
figures to outputs/.

Usage:
    python scripts/03_train.py
    python scripts/03_train.py --data data/processed/analysis_base.parquet  # no-GEE fallback
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
from nfhs_climate.models.evaluate import evaluate, results_table  # noqa: E402
from nfhs_climate.models.features import (  # noqa: E402
    CLIMATE_ACUTE_FEATURES,
    CLIMATE_NORMAL_FEATURES,
    SES_FEATURES,
    assemble,
)
from nfhs_climate.models.models import estimator_factories  # noqa: E402


def _plot_leakage(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["auc_random"], 0.4, label="Random CV (leaky)")
    ax.bar(x + 0.2, table["auc_spatial"], 0.4, label="Spatial CV (honest)")
    ax.set_xticks(x)
    ax.set_xticklabels(table["model"], rotation=15)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Random vs. spatial cross-validation\n(gap = spatial leakage)")
    ax.legend()
    for i, gap in enumerate(table["leakage_gap"]):
        ax.annotate(f"gap {gap:.3f}", (i, table['auc_random'].iloc[i] + 0.01),
                    ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=str(DATA_PROCESSED / "analysis_full.parquet"),
        help="Analysis parquet. Falls back to analysis_base if full is absent.",
    )
    ap.add_argument("--outcome", default="exclusive_hygienic_narrow")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    data_path = Path(args.data)
    if not data_path.exists():
        alt = DATA_PROCESSED / "analysis_base.parquet"
        if alt.exists():
            print(f"{data_path} not found; using {alt} (no acute heat features).")
            data_path = alt
        else:
            print("No analysis parquet found. Run 02a (and ideally 02b) first.")
            return 1

    df = pd.read_parquet(data_path)
    X, y, groups, meta = assemble(df, outcome=args.outcome)
    weight = meta["survey_weight"]

    print(f"\nn={meta['n']:,}  prevalence={meta['prevalence']:.3f}  "
          f"districts={meta['n_districts']}  acute_heat={meta['has_acute']}")

    factories = estimator_factories(meta["categorical"], meta["numeric"])

    # --- Main comparison: each model under random vs spatial CV ---
    pairs = []
    for label, factory in factories.items():
        rnd, spa = evaluate(
            factory, X, y, groups, label, n_splits=args.folds, weight=weight
        )
        pairs.append((rnd, spa))

    table = results_table(pairs)
    table.to_csv(TABLES / "cv_results.csv", index=False)
    print("\n=== Random vs Spatial CV ===")
    print(table.to_string(index=False))
    # _plot_leakage(table, FIGURES / "leakage_gap.png")  # superseded by 05_figures.py:cv_comparison.png

    # --- Ablation: does climate add anything over SES? (spatial CV only) ---
    print("\n=== Feature ablation (spatial CV, gradient boosting) ===")
    blocks = {
        "ses_only": [c for c in SES_FEATURES if c in X.columns],
        "ses_plus_normals": [c for c in SES_FEATURES + CLIMATE_NORMAL_FEATURES if c in X.columns],
    }
    if meta["has_acute"]:
        blocks["ses_plus_all_climate"] = [
            c for c in SES_FEATURES + CLIMATE_NORMAL_FEATURES + CLIMATE_ACUTE_FEATURES
            if c in X.columns
        ]

    abl_rows = []
    for name, cols in blocks.items():
        cat = [c for c in meta["categorical"] if c in cols]
        num = [c for c in meta["numeric"] if c in cols]
        fac = estimator_factories(cat, num)["gradient_boosting"]
        _, spa = evaluate(fac, X[cols], y, groups, name, n_splits=args.folds, weight=weight)
        abl_rows.append(spa.summary())
    abl = pd.DataFrame(abl_rows)[["model", "auc_mean", "auc_std", "ap_mean"]]
    abl.to_csv(TABLES / "ablation.csv", index=False)
    print(abl.to_string(index=False))

    print(f"\nWrote tables to {TABLES} and figure to {FIGURES/'leakage_gap.png'}")
    print("\nHeadline: compare auc_random vs auc_spatial above. The gap is how "
          "much apparent skill was spatial memorisation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
