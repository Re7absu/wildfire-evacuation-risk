"""Survival analysis models: predict WHEN a fire reaches an evacuation zone.

Survival models learn the whole time-to-hit curve from all fires, including
censored ones (a fire last seen at 40 h without a hit still tells us it did not
hit for 40 h). From that curve we read:
  * P(hit by 12 / 24 / 48 / 72 h) = 1 - S(t), where S(t) is the survival function
  * the predicted time to hit (when the chance of a hit passes 50%)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

from src.data import HORIZONS


# Turn the two answer columns into the format scikit-survival expects.
def to_surv(time: np.ndarray | pd.Series, event: np.ndarray | pd.Series) -> np.ndarray:
    """Structured array of (event: bool, time: float) for scikit-survival."""
    t = np.clip(np.asarray(time, dtype=float), 1e-3, None)  # survival models need time > 0
    return Surv.from_arrays(np.asarray(event).astype(bool), t)


# Kaplan-Meier curve: the share of fires that have reached a zone by each hour (handles censoring).
def hit_curve(time: np.ndarray | pd.Series, event: np.ndarray | pd.Series,
              min_at_risk: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return (hours, cumulative share of fires that hit) from the Kaplan-Meier estimator.

    The curve stops once fewer than `min_at_risk` fires are still being watched,
    because a single fire would then move it a lot.
    """
    tt = np.asarray(time, dtype=float)
    t, s = kaplan_meier_estimator(np.asarray(event).astype(bool), tt)
    at_risk = np.array([(tt >= x).sum() for x in t])
    keep = at_risk >= min_at_risk
    return t[keep], 1 - s[keep]


# Wrap any scikit-survival pipeline so it works like the rest of the project (same y, same scorer).
class SurvivalModel(BaseEstimator):
    """A survival pipeline (e.g. Cox PH or Random Survival Forest) with risk-by-horizon outputs.

    `y` is a 2-column array: [time_to_hit_hours, event], the same as MultiHorizonModel.
    """

    # Store the pipeline to train and the time limits to report.
    def __init__(self, estimator: BaseEstimator, horizons: tuple[int, ...] = HORIZONS) -> None:
        self.estimator = estimator
        self.horizons = horizons

    # Train one survival model on all fires, censored ones included.
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SurvivalModel":
        """Fit the survival pipeline on (time, event)."""
        y = np.asarray(y, dtype=float)
        self.model_ = clone(self.estimator).fit(X, to_surv(y[:, 0], y[:, 1]))
        return self

    # Survival curves S(t) for each fire, as step functions from scikit-survival.
    def _curves(self, X: pd.DataFrame) -> list:
        return list(self.model_.predict_survival_function(X))

    # Chance of a hit by each time limit: 1 - S(t). Past the last observed time the curve stays flat.
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(hit by each horizon), shape (n_fires, n_horizons), non-decreasing along each row."""
        out = []
        for fn in self._curves(X):
            end = fn.domain[1]
            out.append([1.0 - float(fn(min(h, end))) for h in self.horizons])
        return np.clip(np.maximum.accumulate(np.array(out), axis=1), 0, 1)

    # Same as predict_proba, but as percentages in a table with named columns.
    def predict_risk(self, X: pd.DataFrame) -> pd.DataFrame:
        """Risk in % per horizon (columns risk_12h_pct, ...)."""
        return pd.DataFrame(self.predict_proba(X) * 100, index=X.index,
                            columns=[f"risk_{h}h_pct" for h in self.horizons])

    # Predicted time to hit: the first hour when the chance of a hit reaches 50% (NaN if never within the data).
    def predict_time(self, X: pd.DataFrame) -> np.ndarray:
        """Median predicted hours until the fire reaches the 5 km zone, or NaN if under 50% throughout."""
        times = []
        for fn in self._curves(X):
            below = np.flatnonzero(fn.y <= 0.5)
            times.append(float(fn.x[below[0]]) if below.size else np.nan)
        return np.array(times)

    # Full curve of P(hit by t) on a grid of hours, for charts.
    def predict_curve(self, X: pd.DataFrame, hours: np.ndarray) -> np.ndarray:
        """P(hit by t) for each fire (rows) at each hour in `hours` (columns)."""
        out = []
        for fn in self._curves(X):
            end = fn.domain[1]
            out.append([1.0 - float(fn(min(max(t, fn.domain[0]), end))) for t in hours])
        return np.array(out)

    # Urgency score from the survival model itself (higher = expected to hit sooner).
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Risk score from the underlying survival model."""
        return self.model_.predict(X)
