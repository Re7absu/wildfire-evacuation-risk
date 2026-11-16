"""Multi-horizon risk model, preprocessing pipeline and the competition metric.

The model is one binary classifier per horizon (12/24/48/72 h). Each
classifier answers "will this fire come within 5 km of an evacuation zone
within *h* hours (counted from t0 + 5 h)?"; the per-horizon probabilities are
then forced to be non-decreasing in *h*. The whole thing is a single
sklearn-compatible estimator, so it works with ``cross_validate``,
``GridSearchCV`` and ``permutation_importance``.

Targets are passed as a 2-column array ``y = [time_to_hit_hours, event]``.

Evaluation follows the WiDS 2026 competition definition::

    Hybrid = 0.3 * C-index + 0.7 * (1 - Weighted Brier)
    Weighted Brier = 0.3 * Brier@24h + 0.4 * Brier@48h + 0.3 * Brier@72h

where each Brier@H is censor-aware (rows censored before H are excluded).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import HORIZONS, censor_aware_labels, horizon_labels
from src.features import FeatureEngineer

BRIER_WEIGHTS: dict[int, float] = {24: 0.3, 48: 0.4, 72: 0.3}
HYBRID_WEIGHTS: dict[str, float] = {"c_index": 0.3, "brier": 0.7}
HEADLINE_HORIZON: int = 48  # the most heavily weighted, "strongest operational value" horizon

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
        label_mode: How censored fires are labelled for training:

            * ``"censor_aware"`` (competition rule): a fire censored before
              ``H`` is dropped from the ``H`` model's training rows.
            * ``"metadata"``: censored fires are negatives at every horizon
              (the data dictionary's "never hit within 72 h" reading).

            When a horizon's training labels contain a single class (e.g. at
            72 h under ``censor_aware`` every evaluable fire is a hit), the
            model predicts the Laplace-smoothed base rate ``c = (k + 1) / (n + 2)``
            plus a rank-preserving term, ``c + (1 - c) * mean(earlier horizons)``,
            so that column still orders fires by urgency (the C-index is then
            the same whichever horizon column a grader ranks on).
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        horizons: tuple[int, ...] = HORIZONS,
        label_mode: str = "censor_aware",
    ) -> None:
        self.estimator = estimator
        self.horizons = horizons
        self.label_mode = label_mode

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.DataFrame) -> "MultiHorizonRiskModel":
        """Fit one model per horizon.

        Args:
            X: Feature frame.
            y: Array-like of shape (n, 2): ``[time_to_hit_hours, event]``.

        Returns:
            ``self``.
        """
        if self.label_mode not in ("censor_aware", "metadata"):
            raise ValueError(f"unknown label_mode {self.label_mode!r}")
        y_arr = np.asarray(y, dtype=float)
        label_fn = censor_aware_labels if self.label_mode == "censor_aware" else horizon_labels
        labels = label_fn(y_arr[:, 0], y_arr[:, 1], tuple(self.horizons))
        X = X.reset_index(drop=True)
        self.models_: dict[int, Any] = {}
        for h in self.horizons:
            yh = labels[f"hit_{h}h"].to_numpy(dtype=float)
            keep = ~np.isnan(yh)
            yk = yh[keep].astype(int)
            if yk.min() == yk.max():  # single class: smoothed base rate
                self.models_[h] = float((yk.sum() + 1) / (len(yk) + 2))
            else:
                self.models_[h] = clone(self.estimator).fit(X.loc[keep], yk)
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of a hit within each horizon, shape (n, n_horizons).

        Columns follow ``self.horizons`` and are non-decreasing along each row.
        """
        cols: list[np.ndarray] = []
        for h in self.horizons:
            m = self.models_[h]
            if not isinstance(m, float):
                cols.append(m.predict_proba(X)[:, 1])
            elif cols:
                # Single-class horizon (e.g. 72 h under censor-aware labels): keep the
                # level at the base rate c, but order fires by the earlier horizons so
                # the column stays rank-informative: c + (1 - c) * mean(earlier probs).
                cols.append(m + (1.0 - m) * np.mean(cols, axis=0))
            else:
                cols.append(np.full(len(X), m))
        return np.clip(np.maximum.accumulate(np.column_stack(cols), axis=1), 0.0, 1.0)

    def predict_risk(self, X: pd.DataFrame) -> pd.DataFrame:
        """Risk percentage per horizon as a frame with columns ``risk_<h>h_pct``."""
        p = self.predict_proba(X) * 100
        return pd.DataFrame(p, columns=[f"risk_{h}h_pct" for h in self.horizons], index=X.index)


# --------------------------------------------------------------------------- #
# Competition metric
# --------------------------------------------------------------------------- #
def risk_score(probs: np.ndarray) -> np.ndarray:
    """Urgency score used for the C-index: mean of the horizon probabilities.

    The mean of P(hit by 12/24/48/72 h) is a discrete area under the predicted
    cumulative-incidence curve: fires expected to hit *sooner* score higher.
    (The competition text does not state which score its C-index uses; this is
    our documented assumption.)
    """
    return np.asarray(probs, dtype=float).mean(axis=1)


def concordance_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C-index for right-censored data.

    A pair (i, j) is comparable when fire i hit (event=1) strictly before
    j's observed time (j may be a hit or censored). It is concordant when
    ``risk_i > risk_j``; ties in risk count 0.5.

    Args:
        risk: Higher = more urgent.
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


def censored_brier(prob: np.ndarray, time: np.ndarray, event: np.ndarray, horizon: int) -> float:
    """Censor-aware Brier score at one horizon (competition rule).

    Hits -> label 1 if hit by ``horizon`` else 0; censored after the horizon ->
    0; censored at/before the horizon -> excluded.

    Returns:
        Mean squared error over evaluable rows, or NaN if none.
    """
    lab = censor_aware_labels(time, event, (horizon,))[f"hit_{horizon}h"].to_numpy()
    keep = ~np.isnan(lab)
    if not keep.any():
        return float("nan")
    return float(np.mean((np.asarray(prob, dtype=float)[keep] - lab[keep]) ** 2))


def competition_scores(probs: np.ndarray, time: np.ndarray, event: np.ndarray,
                       horizons: tuple[int, ...] = HORIZONS) -> dict[str, float]:
    """All competition metrics from a probability matrix.

    Args:
        probs: Shape (n, len(horizons)) of P(hit by h).
        time: ``time_to_hit_hours``.
        event: Event indicator.
        horizons: Horizons matching the columns of ``probs``.

    Returns:
        Dict with ``brier_<h>h``, ``auc_<h>h`` (censor-aware, NaN if one class),
        ``weighted_brier``, ``c_index`` and ``hybrid``.
    """
    probs = np.asarray(probs, dtype=float)
    time, event = np.asarray(time, dtype=float), np.asarray(event, dtype=float)
    labels = censor_aware_labels(time, event, tuple(horizons))
    out: dict[str, float] = {}
    for k, h in enumerate(horizons):
        out[f"brier_{h}h"] = censored_brier(probs[:, k], time, event, h)
        lab = labels[f"hit_{h}h"].to_numpy()
        keep = ~np.isnan(lab)
        yk = lab[keep]
        out[f"auc_{h}h"] = roc_auc_score(yk, probs[keep, k]) if 0 < yk.sum() < len(yk) else float("nan")
    out["weighted_brier"] = float(sum(w * out[f"brier_{h}h"] for h, w in BRIER_WEIGHTS.items()))
    out["c_index"] = concordance_index(risk_score(probs), time, event)
    out["hybrid"] = HYBRID_WEIGHTS["c_index"] * out["c_index"] + HYBRID_WEIGHTS["brier"] * (1 - out["weighted_brier"])
    return out


def horizon_scores(model: MultiHorizonRiskModel, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    """Competition metrics for a fitted model on ``(X, y)``; see :func:`competition_scores`."""
    y_arr = np.asarray(y, dtype=float)
    return competition_scores(model.predict_proba(X), y_arr[:, 0], y_arr[:, 1], tuple(model.horizons))


def hybrid_scorer(model: MultiHorizonRiskModel, X: pd.DataFrame, y: np.ndarray) -> float:
    """Scorer (higher is better): the competition Hybrid score."""
    return horizon_scores(model, X, y)["hybrid"]


def neg_weighted_brier(model: MultiHorizonRiskModel, X: pd.DataFrame, y: np.ndarray) -> float:
    """Scorer (higher is better): minus the weighted censor-aware Brier score."""
    return -horizon_scores(model, X, y)["weighted_brier"]


def c_index_scorer(model: MultiHorizonRiskModel, X: pd.DataFrame, y: np.ndarray) -> float:
    """Scorer: Harrell's C-index of :func:`risk_score`."""
    return horizon_scores(model, X, y)["c_index"]


SCORING: dict[str, Any] = {"hybrid": hybrid_scorer, "neg_weighted_brier": neg_weighted_brier, "c_index": c_index_scorer}


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
