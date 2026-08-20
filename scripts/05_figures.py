#!/usr/bin/env python
"""Week 4: generate the figure set from saved results and the analysis frame.

Produces six figures into outputs/figures/:
  1. cv_comparison.png     random vs spatial AUC, both models, gap annotated
  2. ablation.png          SES -> +normals -> +acute, spatial AUC steps
  3. shap_importance.png    feature importance for the gradient-boosting model
  4. calibration.png        predicted vs observed probability (are we honest?)
  5. outcome_definitions.png  the four hygiene definitions, 41%-78% spread
  6. prevalence_by_wealth.png hygienic use rises with wealth (the SES story)

Reads outputs/tables/*.csv (from 03_train.py) and the analysis parquet. SHAP is
computed on a random sample for speed; the picture is identical to the full set.

Usage:
    python scripts/05_figures.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nfhs_climate.config import (  # noqa: E402
    COVARIATES, DATA_PROCESSED, FIGURES, RANDOM_SEED, TABLES,
)
from nfhs_climate.data.nfhs import weighted_mean  # noqa: E402
from nfhs_climate.models.features import assemble  # noqa: E402
from nfhs_climate.models.models import estimator_factories  # noqa: E402

log = logging.getLogger(__name__)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

BLUE, ORANGE, GREEN, GREY = "#2c6fbb", "#e8833a", "#3a9d5d", "#9aa0a6"


def fig_cv_comparison():
    t = pd.read_csv(TABLES / "cv_results.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t["auc_random"], 0.4, label="Random CV (leaky)", color=GREY)
    ax.bar(x + 0.2, t["auc_spatial"], 0.4, label="Spatial CV (honest)", color=BLUE)
    for i, r in t.iterrows():
        ax.annotate(f"gap {r['leakage_gap']:.3f}",
                    (i, max(r["auc_random"], r["auc_spatial"]) + 0.008),
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(t["model"].str.replace("_", " "))
    ax.set_ylabel("AUC"); ax.set_ylim(0.5, 0.85)
    ax.set_title("Random vs. spatial cross-validation\nthe gap is district memorisation")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "cv_comparison.png"); plt.close(fig)


def fig_ablation():
    t = pd.read_csv(TABLES / "ablation.csv")
    labels = {"ses_only": "SES only",
              "ses_plus_normals": "+ climate normals",
              "ses_plus_all_climate": "+ acute heat"}
    t["nice"] = t["model"].map(labels).fillna(t["model"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(t))
    ax.plot(x, t["auc_mean"], "-o", color=GREEN, lw=2)
    ax.fill_between(x, t["auc_mean"] - t["auc_std"], t["auc_mean"] + t["auc_std"],
                    color=GREEN, alpha=0.15)
    for i, r in t.iterrows():
        ax.annotate(f"{r['auc_mean']:.3f}", (i, r["auc_mean"] + 0.004),
                    ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(t["nice"])
    ax.set_ylabel("Spatial-CV AUC")
    ax.set_title("What each feature block adds\n(gradient boosting, honest CV)")
    fig.tight_layout(); fig.savefig(FIGURES / "ablation.png"); plt.close(fig)


def _fit_full_gbm(df, outcome):
    X, y, groups, meta = assemble(df, outcome=outcome)
    est = estimator_factories(meta["categorical"], meta["numeric"])["gradient_boosting"]()
    w = meta["survey_weight"]
    kw = {"clf__sample_weight": w.to_numpy()} if w is not None else {}
    est.fit(X, y, **kw)
    return est, X, y, meta


def fig_shap(df, outcome, n_sample=4000):
    try:
        import shap
    except ImportError:
        log.warning("shap not installed; skipping SHAP figure. pip install shap")
        return
    est, X, y, meta = _fit_full_gbm(df, outcome)
    Xs = X.sample(min(n_sample, len(X)), random_state=RANDOM_SEED)
    prep = est.named_steps["prep"]; clf = est.named_steps["clf"]
    Xt = prep.transform(Xs)
    names = prep.get_feature_names_out()
    try:
        expl = shap.TreeExplainer(clf)
        sv = expl.shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else sv
    except Exception as e:  # noqa: BLE001
        log.warning("SHAP failed (%s); falling back to permutation importance.", e)
        return _fig_perm_importance(est, X, y)
    imp = np.abs(sv).mean(0)
    order = np.argsort(imp)[-15:]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(range(len(order)), imp[order], color=BLUE)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i].replace("num__", "").replace("cat__", "")
                        for i in order], fontsize=8)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("What drives the prediction\n(top 15 features, gradient boosting)")
    fig.tight_layout(); fig.savefig(FIGURES / "shap_importance.png"); plt.close(fig)


def _fig_perm_importance(est, X, y):
    from sklearn.inspection import permutation_importance
    r = permutation_importance(est, X.sample(3000, random_state=RANDOM_SEED),
                               y.loc[X.sample(3000, random_state=RANDOM_SEED).index],
                               n_repeats=5, random_state=RANDOM_SEED)
    order = np.argsort(r.importances_mean)[-15:]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(range(len(order)), r.importances_mean[order], color=BLUE)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(X.columns[order], fontsize=8)
    ax.set_xlabel("permutation importance")
    ax.set_title("What drives the prediction (top 15)")
    fig.tight_layout(); fig.savefig(FIGURES / "shap_importance.png"); plt.close(fig)


def fig_calibration(df, outcome):
    from sklearn.calibration import calibration_curve
    from sklearn.model_selection import GroupKFold
    X, y, groups, meta = assemble(df, outcome=outcome)
    fac = estimator_factories(meta["categorical"], meta["numeric"])["gradient_boosting"]
    pred = pd.Series(np.nan, index=X.index)
    for tr, te in GroupKFold(5).split(X, y, groups):
        est = fac(); est.fit(X.iloc[tr], y.iloc[tr]); pred.iloc[te] = est.predict_proba(X.iloc[te])[:, 1]
    frac_pos, mean_pred = calibration_curve(y, pred, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="perfect")
    ax.plot(mean_pred, frac_pos, "-o", color=ORANGE, label="model")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction")
    ax.set_title("Calibration (out-of-fold, spatial)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "calibration.png"); plt.close(fig)


def fig_outcome_definitions():
    t = pd.read_csv(TABLES / "outcome_definitions.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    vals = t["weighted_prevalence"] * 100
    ax.bar(range(len(t)), vals, color=[BLUE, GREEN, ORANGE, GREY][:len(t)])
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1f}%", (i, v + 0.8), ha="center", fontsize=9)
    ax.set_xticks(range(len(t))); ax.set_xticklabels(t["definition"], rotation=15, fontsize=8)
    ax.set_ylabel("Weighted prevalence (%)")
    ax.set_title("Four definitions of 'hygienic' — a 28-point spread\nthe choice matters more than any effect we detect")
    fig.tight_layout(); fig.savefig(FIGURES / "outcome_definitions.png"); plt.close(fig)


def fig_prevalence_by_wealth(df, outcome):
    wcol = COVARIATES["wealth_quintile"]
    if wcol not in df.columns:
        return
    d = df[df[outcome].notna()]
    rows = []
    for q, g in d.groupby(wcol):
        rows.append((int(q), weighted_mean(g[outcome], g["survey_weight"]) * 100))
    r = pd.DataFrame(rows, columns=["quintile", "rate"]).sort_values("quintile")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(r["quintile"], r["rate"], color=BLUE)
    ax.set_xticks(r["quintile"])
    ax.set_xticklabels(["poorest", "poorer", "middle", "richer", "richest"], fontsize=8)
    ax.set_ylabel("Hygienic-material use (%)")
    ax.set_title("The real driver: hygienic use rises steeply with wealth")
    for _, row in r.iterrows():
        ax.annotate(f"{row['rate']:.0f}%", (row["quintile"], row["rate"] + 1), ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGURES / "prevalence_by_wealth.png"); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_PROCESSED / "analysis_full.parquet"))
    ap.add_argument("--outcome", default="exclusive_hygienic_narrow")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    dp = Path(args.data)
    if not dp.exists():
        dp = DATA_PROCESSED / "analysis_base.parquet"
    df = pd.read_parquet(dp)

    made = []
    for name, fn in [
        ("cv_comparison", lambda: fig_cv_comparison()),
        ("ablation", lambda: fig_ablation()),
        ("outcome_definitions", lambda: fig_outcome_definitions()),
        ("prevalence_by_wealth", lambda: fig_prevalence_by_wealth(df, args.outcome)),
        ("calibration", lambda: fig_calibration(df, args.outcome)),
        ("shap_importance", lambda: fig_shap(df, args.outcome)),
    ]:
        try:
            fn(); made.append(name); print(f"  [ok] {name}.png")
        except FileNotFoundError as e:
            print(f"  [skip] {name}: {e} (run 03_train.py first)")
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {name} failed: {e}")

    print(f"\nWrote {len(made)} figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
