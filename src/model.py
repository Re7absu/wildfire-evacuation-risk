"""The model (one classifier per horizon) and the competition score."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import HORIZONS, make_labels
from src.features import FeatureAdder

MAIN_HORIZON = 48  # the risk we show users (the competition weights 48 h most)
BANDS = ["Low", "Medium", "High", "Critical"]  # 0-25, 25-50, 50-75, 75-100 %


# Build the full processing chain: add features -> clean/scale -> model.
def make_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Add features -> fill missing values + scale -> classifier."""
    # Number columns: fill gaps with the median, then put them on the same scale.
    numeric = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    # Text columns (none in this data): fill gaps, then turn categories into 0/1 columns.
    text = Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    # Send each column to the right branch based on its type.
    prep = ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include="number")),
        ("cat", text, make_column_selector(dtype_include=["object", "category"])),
    ])
    return Pipeline([("features", FeatureAdder()), ("prep", prep), ("clf", classifier)])


# One model per time limit (12/24/48/72 h), wrapped as a single sklearn model.
class MultiHorizonModel(BaseEstimator):
    """Trains one copy of the pipeline per horizon (12, 24, 48, 72 h).

    `y` is a 2-column array: [time_to_hit_hours, event].
    """

    # Store the settings: which pipeline to copy, and which time limits to use.
    def __init__(self, estimator: BaseEstimator, horizons: tuple[int, ...] = HORIZONS) -> None:
        self.estimator = estimator
        self.horizons = horizons

    # Train a separate copy of the pipeline for each time limit.
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MultiHorizonModel":
        """Fit one model per horizon, skipping rows with an unknown label."""
        y = np.asarray(y, dtype=float)
        labels = make_labels(y[:, 0], y[:, 1], self.horizons)
        X = X.reset_index(drop=True)
        self.models_ = {}
        for h in self.horizons:
            lab = labels[f"hit_{h}h"]
            known = lab.notna()  # skip fires whose outcome at this time limit is unknown
            if lab[known].nunique() == 1:
                # Only one class (at 72 h every known fire is a hit): use the average rate.
                self.models_[h] = (lab[known].sum() + 1) / (known.sum() + 2)
            else:
                self.models_[h] = clone(self.estimator).fit(X[known], lab[known].astype(int))
        return self

    # Predict the chance (0 to 1) of a hit by each time limit.
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Chance of a hit by each horizon. Each column is >= the one before it."""
        cols = []
        for h in self.horizons:
            m = self.models_[h]
            if isinstance(m, float):
                # Same level for everyone, but still higher for more urgent fires.
                cols.append(m + (1 - m) * np.mean(cols, axis=0) if cols else np.full(len(X), m))
            else:
                cols.append(m.predict_proba(X)[:, 1])
        # Risk can only go up with more time, so make each column at least the one before.
        return np.clip(np.maximum.accumulate(np.column_stack(cols), axis=1), 0, 1)

    # Same as predict_proba, but as percentages in a table with named columns.
    def predict_risk(self, X: pd.DataFrame) -> pd.DataFrame:
        """Risk in % per horizon (columns risk_12h_pct, ...)."""
        return pd.DataFrame(self.predict_proba(X) * 100, index=X.index,
                            columns=[f"risk_{h}h_pct" for h in self.horizons])


# ---------------- Competition score ----------------
# Hybrid = 0.3 * C-index + 0.7 * (1 - Brier),   Brier = 0.3*B24 + 0.4*B48 + 0.3*B72

# Ranking score: how often a fire that hit sooner got a higher risk than a later one.
def c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Share of fire pairs ranked in the right order (fire that hit sooner = higher risk)."""
    risk, time, event = (np.asarray(a, dtype=float) for a in (risk, time, event))
    # A pair counts when fire i hit before fire j's time.
    pairs = (event[:, None] == 1) & (time[:, None] < time[None, :])
    diff = risk[:, None] - risk[None, :]
    # Right order = 1 point, a tie = half a point.
    return float(((diff > 0) + 0.5 * (diff == 0))[pairs].mean())


# Accuracy of the percentages at one time limit (lower is better, 0 is perfect).
def brier(prob: np.ndarray, time: np.ndarray, event: np.ndarray, h: int) -> float:
    """Mean squared error of the probabilities at one horizon, skipping unknown rows."""
    lab = make_labels(time, event, (h,))[f"hit_{h}h"].to_numpy()
    known = ~np.isnan(lab)
    return float(np.mean((np.asarray(prob)[known] - lab[known]) ** 2))


# Compute every competition number at once: Brier per time limit, C-index and Hybrid.
def score(probs: np.ndarray, time: np.ndarray, event: np.ndarray) -> dict[str, float]:
    """All competition numbers for a (n_fires, 4) probability matrix."""
    s = {f"brier_{h}h": brier(probs[:, k], time, event, h) for k, h in enumerate(HORIZONS)}
    s["brier"] = 0.3 * s["brier_24h"] + 0.4 * s["brier_48h"] + 0.3 * s["brier_72h"]
    s["c_index"] = c_index(probs.mean(axis=1), time, event)
    s["hybrid"] = 0.3 * s["c_index"] + 0.7 * (1 - s["brier"])
    return s


# Scoring function in the format sklearn expects (used by cross-validation and tuning).
def hybrid_scorer(model: MultiHorizonModel, X: pd.DataFrame, y: np.ndarray) -> float:
    """sklearn scorer: the Hybrid score (higher is better)."""
    y = np.asarray(y, dtype=float)
    return score(model.predict_proba(X), y[:, 0], y[:, 1])["hybrid"]


# Turn a risk percentage into a simple label for decision makers.
def risk_band(pct: np.ndarray | pd.Series) -> pd.Series:
    """Turn a risk % into Low / Medium / High / Critical."""
    return pd.Series(pd.cut(np.asarray(pct, dtype=float), [-1, 25, 50, 75, 101], labels=BANDS, right=False))
