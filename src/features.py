"""Feature engineering for the fire-risk model.

All engineered features are deterministic row-wise transforms of the raw
columns, wrapped in an sklearn transformer so they live inside the saved
pipeline and are applied identically at training and prediction time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

HIT_RADIUS_M: float = 5_000.0  # the target is "fire within 5 km of an evac zone"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physically motivated features to a copy of ``df``.

    New columns:
        * ``log_dist_min``     - log1p of minimum distance to evac zone (m); distance
          is extremely right-skewed (hundreds of m to >700 km).
        * ``dist_minus_radius_km`` - distance left before the 5 km hit radius (km, >= 0).
        * ``within_10km``      - flag: fire already within 10 km of an evac zone.
        * ``eta_hours``        - naive time to reach the 5 km radius at the current
          closing speed (capped at 1000 h; 1000 when not closing).
        * ``has_growth``       - flag: any measurable area growth in the first 5 h.
        * ``hour_sin/hour_cos``, ``month_sin/month_cos`` - cyclic time encodings.

    Args:
        df: Frame with the raw feature columns.

    Returns:
        Copy of ``df`` with the extra columns appended.
    """
    out = df.copy()
    dist = out["dist_min_ci_0_5h"].clip(lower=0)
    remaining = (dist - HIT_RADIUS_M).clip(lower=0)
    out["log_dist_min"] = np.log1p(dist)
    out["dist_minus_radius_km"] = remaining / 1000.0
    out["within_10km"] = (dist <= 10_000).astype(int)
    closing = out["closing_speed_m_per_h"]
    eta = np.where(closing > 1e-6, remaining / closing.where(closing > 1e-6, 1.0), 1000.0)
    out["eta_hours"] = np.clip(eta, 0, 1000.0)
    out["has_growth"] = (out["area_growth_abs_0_5h"] > 0).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["event_start_hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["event_start_hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["event_start_month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["event_start_month"] / 12)
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn transformer wrapper around :func:`add_engineered_features`."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "FeatureEngineer":
        """No-op fit (the transform is stateless). Returns ``self``."""
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X`` with engineered columns added."""
        return add_engineered_features(pd.DataFrame(X))

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Names of the output columns (raw inputs + engineered)."""
        dummy = pd.DataFrame(np.zeros((1, len(self.feature_names_in_))), columns=self.feature_names_in_)
        return np.asarray(add_engineered_features(dummy).columns, dtype=object)
