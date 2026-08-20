"""Model evaluation under random vs. spatial (district-grouped) CV.

This is the heart of the project. The claim:

    Random k-fold cross-validation OVERSTATES how well a model predicts
    menstrual hygiene practice, because test-set women live in the same
    districts as training-set women. The model learns district-level baselines
    and gets credit for "predicting" them. Grouping folds by district -- so
    every test district is unseen in training -- removes that credit and gives
    the honest estimate of how the model generalises to new places.

The gap between the two AUCs is the finding. A large gap means most of the
apparent skill was spatial memorisation, not transferable signal. Reporting
the smaller, honest number is the point of the whole exercise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import RANDOM_SEED

log = logging.getLogger(__name__)


def make_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    """Impute + encode. Median for numeric, most-frequent + one-hot for cat."""
    num = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    cat = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", num, numeric), ("cat", cat, categorical)],
        remainder="drop",
        sparse_threshold=0.0,
    )


@dataclass
class CVResult:
    label: str
    scheme: str            # "random" or "spatial"
    auc: list[float] = field(default_factory=list)
    ap: list[float] = field(default_factory=list)
    brier: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        def ms(x):
            return (float(np.mean(x)), float(np.std(x)))
        a_m, a_s = ms(self.auc)
        p_m, p_s = ms(self.ap)
        b_m, b_s = ms(self.brier)
        return {
            "model": self.label,
            "cv_scheme": self.scheme,
            "auc_mean": a_m, "auc_std": a_s,
            "ap_mean": p_m, "ap_std": p_s,
            "brier_mean": b_m, "brier_std": b_s,
            "n_folds": len(self.auc),
        }


def _fold_scores(
    estimator, X, y, train_idx, test_idx, weight=None
) -> tuple[float, float, float]:
    Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
    ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
    fit_kw = {}
    if weight is not None:
        # survey weights on the training fold only
        fit_kw["clf__sample_weight"] = weight.iloc[train_idx].to_numpy()
    estimator.fit(Xtr, ytr, **fit_kw)
    p = estimator.predict_proba(Xte)[:, 1]
    return (
        roc_auc_score(yte, p),
        average_precision_score(yte, p),
        brier_score_loss(yte, p),
    )


def evaluate(
    make_estimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    label: str,
    n_splits: int = 5,
    weight: pd.Series | None = None,
) -> tuple[CVResult, CVResult]:
    """Run the SAME model under random and spatial CV. Returns (random, spatial).

    make_estimator: zero-arg callable returning a fresh unfitted Pipeline, so
    each fold gets a clean model.
    """
    rng = RANDOM_SEED

    # Random: stratified k-fold, ignores geography -- the leaky baseline
    rnd = CVResult(label, "random")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng)
    for tr, te in skf.split(X, y):
        a, p, b = _fold_scores(make_estimator(), X, y, tr, te, weight)
        rnd.auc.append(a); rnd.ap.append(p); rnd.brier.append(b)

    # Spatial: group k-fold on district -- every test district unseen in train
    spa = CVResult(label, "spatial")
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        a, p, b = _fold_scores(make_estimator(), X, y, tr, te, weight)
        spa.auc.append(a); spa.ap.append(p); spa.brier.append(b)

    log.info(
        "%-18s random AUC %.3f  |  spatial AUC %.3f  |  leakage gap %.3f",
        label,
        np.mean(rnd.auc), np.mean(spa.auc),
        np.mean(rnd.auc) - np.mean(spa.auc),
    )
    return rnd, spa


def results_table(pairs: list[tuple[CVResult, CVResult]]) -> pd.DataFrame:
    """Tidy table with the leakage gap made explicit."""
    rows = []
    for rnd, spa in pairs:
        r, s = rnd.summary(), spa.summary()
        rows.append(
            {
                "model": r["model"],
                "auc_random": r["auc_mean"],
                "auc_spatial": s["auc_mean"],
                "leakage_gap": r["auc_mean"] - s["auc_mean"],
                "ap_spatial": s["ap_mean"],
                "brier_spatial": s["brier_mean"],
                "auc_spatial_std": s["auc_std"],
            }
        )
    return pd.DataFrame(rows)
