"""Model definitions: a logistic baseline and a gradient-boosting model.

Two models, deliberately:
- Logistic regression: interpretable, linear baseline. If a fancy model can't
  beat this, that is itself the finding.
- Gradient boosting (LightGBM if available, else sklearn HistGB): captures
  interactions and non-linearities, the realistic "good" model.

Both are wrapped in the same preprocessing pipeline so the random-vs-spatial
comparison is apples to apples.
"""

from __future__ import annotations

import logging

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..config import RANDOM_SEED
from .evaluate import make_preprocessor

log = logging.getLogger(__name__)


def _has_lightgbm() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:   # ImportError OR OSError (missing libomp on macOS)
        return False


def make_logistic(categorical, numeric):
    """Fresh logistic-regression pipeline."""
    return Pipeline(
        [
            ("prep", make_preprocessor(categorical, numeric)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def make_gbm(categorical, numeric):
    """Fresh gradient-boosting pipeline (LightGBM or sklearn fallback)."""
    if _has_lightgbm():
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            verbose=-1,
        )
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier

        log.info("LightGBM not installed; using sklearn HistGradientBoosting.")
        clf = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.03,
            max_leaf_nodes=31,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
    return Pipeline(
        [("prep", make_preprocessor(categorical, numeric)), ("clf", clf)]
    )


def estimator_factories(categorical, numeric):
    """Return {label: zero-arg factory} so each CV fold gets a fresh model."""
    return {
        "logistic": lambda: make_logistic(categorical, numeric),
        "gradient_boosting": lambda: make_gbm(categorical, numeric),
    }
