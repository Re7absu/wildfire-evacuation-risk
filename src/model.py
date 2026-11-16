"""Multi-horizon risk model, preprocessing pipeline and evaluation metrics.

The model is one binary classifier per horizon (12/24/48/72 h). Each
classifier answers "will this fire come within 5 km of an evacuation zone
within *h* hours?"; the per-horizon probabilities are then forced to be
non-decreasing in *h* (a fire cannot be less likely to have hit by 48 h than
by 24 h). The whole thing is a single sklearn-compatible estimator, so it works
with ``cross_validate``, ``GridSearchCV`` and ``permutation_importance``.

Targets are passed as a 2-column array ``y = [time_to_hit_hours, event]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import HORIZONS, horizon_labels
from src.features import FeatureEngineer

RISK_BANDS: list[tuple[str, float, float]] = [
    ("Low", 0.0, 25.0),
    ("Medium", 25.0, 50.0),
    ("High", 50.0, 75.0),
    ("Critical", 75.0, 100.0),
]


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #
def make_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer used inside every per-horizon pipeline.

    * numeric columns  -> ``SimpleImputer(median)`` -> ``StandardScaler``
    * categorical cols -> ``SimpleImputer(most_frequent)`` -> ``OneHotEncoder``

    Columns are selected by dtype, so the transformer adapts to whatever the
    feature-engineering step produces (this dataset has no categorical columns,
    so that branch is simply empty).

    Returns:
        Unfitted ``ColumnTransformer``.
    """
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, make_column_selector(dtype_include="number")),
            ("cat", categorical, make_column_selector(dtype_include=["object", "category"])),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Feature engineering -> preprocessing -> classifier.

    Args:
        classifier: Any sklearn classifier with ``predict_proba``.

    Returns:
        Unfitted ``Pipeline`` with steps ``features``, ``prep``, ``clf``.
    """
    return Pipeline([("features", FeatureEngineer()), ("prep", make_preprocessor()), ("clf", classifier)])


# --------------------------------------------------------------------------- #
# Multi-horizon estimator
# --------------------------------------------------------------------------- #
class MultiHorizonRiskModel(BaseEstimator):
    """One cloned pipeline per horizon, with monotone (non-decreasing) outputs.

    Args:
        estimator: Template pipeline/classifier, cloned for each horizon.
        horizons: Horizons in hours.
    """

    def __init__(self, estimator: BaseEstimator, horizons: tuple[int, ...] = HORIZONS) -> None:
        self.estimator = estimator
        self.horizons = horizons

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.DataFrame) -> "MultiHorizonRiskModel":
        """Fit one model per horizon.

        Args:
            X: Feature frame.
            y: Array-like of shape (n, 2): ``[time_to_hit_hours, event]``.

        Returns:
            ``self``.
        """
        y_arr = np.asarray(y, dtype=float)
        labels = horizon_labels(y_arr[:, 0], y_arr[:, 1], tuple(self.horizons))
        self.models_: dict[int, Any] = {}
        for h in self.horizons:
            yh = labels[f"hit_{h}h"].to_numpy()
            if yh.min() == yh.max():  # degenerate fold: constant prediction
                self.models_[h] = float(yh[0])
            else:
                self.models_[h] = clone(self.estimator).fit(X, yh)
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of a hit within each horizon, shape (n, n_horizons).

        Columns follow ``self.horizons`` and are non-decreasing along each row.
        """
        cols = []
        for h in self.horizons:
            m = self.models_[h]
            cols.append(np.full(len(X), m) if isinstance(m, float) else m.predict_proba(X)[:, 1])
        return np.maximum.accumulate(np.column_stack(cols), axis=1)

    def predict_risk(self, X: pd.DataFrame) -> pd.DataFrame:
        """Risk percentage per horizon as a frame with columns ``risk_<h>h_pct``."""
        p = self.predict_proba(X) * 100
        return pd.DataFrame(p, columns=[f"risk_{h}h_pct" for h in self.horizons], index=X.index)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def concordance_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C-index: share of comparable pairs ranked correctly.

    A pair (i, j) is comparable when fire i hit (event=1) earlier than j's time.
    It is concordant when ``risk_i > risk_j``; ties in risk count 0.5.

    Args:
        risk: Higher = more at risk (e.g. mean predicted probability).
        time: ``time_to_hit_hours``.
        event: Event indicator.

    Returns:
        C-index in [0, 1] (0.5 = random), or NaN if no comparable pairs.
    """
    risk, time, event = (np.asarray(a, dtype=float) for a in (risk, time, event))
    comp = (event[:, None] == 1) & (time[:, None] < time[None, :])
    if not comp.any():
        return float("nan")
    diff = risk[:, None] - risk[None, :]
    score = (diff > 0) + 0.5 * (diff == 0)
    return float(score[comp].sum() / comp.sum())


def horizon_scores(model: MultiHorizonRiskModel, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    """Brier, ROC-AUC per horizon plus their means and the C-index.

    Args:
        model: Fitted :class:`MultiHorizonRiskModel`.
        X: Features.
        y: ``[time, event]`` array.

    Returns:
        Dict of metric name -> value.
    """
    y_arr = np.asarray(y, dtype=float)
    probs = model.predict_proba(X)
    labels = horizon_labels(y_arr[:, 0], y_arr[:, 1], tuple(model.horizons))
    out: dict[str, float] = {}
    for k, h in enumerate(model.horizons):
        yh = labels[f"hit_{h}h"].to_numpy()
        out[f"brier_{h}h"] = brier_score_loss(yh, probs[:, k])
        out[f"auc_{h}h"] = roc_auc_score(yh, probs[:, k]) if 0 < yh.sum() < len(yh) else np.nan
    out["mean_brier"] = float(np.mean([out[f"brier_{h}h"] for h in model.horizons]))
    out["mean_auc"] = float(np.nanmean([out[f"auc_{h}h"] for h in model.horizons]))
    out["c_index"] = concordance_index(probs.mean(axis=1), y_arr[:, 0], y_arr[:, 1])
    return out


def neg_mean_brier(model: MultiHorizonRiskModel, X: pd.DataFrame, y: np.ndarray) -> float:
    """Scorer (higher is better): minus the mean Brier score across horizons."""
    return -horizon_scores(model, X, y)["mean_brier"]


def mean_auc(model: MultiHorizonRiskModel, X: pd.DataFrame, y: np.ndarray) -> float:
    """Scorer: mean ROC-AUC across horizons."""
    return horizon_scores(model, X, y)["mean_auc"]


def c_index_scorer(model: MultiHorizonRiskModel, X: pd.DataFrame, y: np.ndarray) -> float:
    """Scorer: Harrell's C-index of the mean predicted probability."""
    return horizon_scores(model, X, y)["c_index"]


SCORING: dict[str, Any] = {"neg_mean_brier": neg_mean_brier, "mean_auc": mean_auc, "c_index": c_index_scorer}


def risk_band(pct: np.ndarray | pd.Series) -> pd.Series:
    """Map risk percentages (0-100) to Low / Medium / High / Critical.

    Args:
        pct: Risk percentages.

    Returns:
        Categorical series of band names (ordered Low < ... < Critical).
    """
    labels = [b[0] for b in RISK_BANDS]
    edges = [0.0, 25.0, 50.0, 75.0, 100.0 + 1e-9]
    return pd.Series(
        pd.Categorical(
            pd.cut(np.asarray(pct, dtype=float), bins=edges, labels=labels, right=False, include_lowest=True),
            categories=labels,
            ordered=True,
        )
    )
