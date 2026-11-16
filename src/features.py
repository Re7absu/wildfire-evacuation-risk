"""Extra features added before training."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few simple distance and time features."""
    out = df.copy()
    dist = out["dist_min_ci_0_5h"].clip(lower=0)            # metres to the evacuation zone
    left = (dist - 5000).clip(lower=0)                      # metres left before the 5 km line
    speed = out["closing_speed_m_per_h"]
    out["log_dist_min"] = np.log1p(dist)
    out["dist_left_km"] = left / 1000
    out["within_10km"] = (dist <= 10_000).astype(int)
    out["eta_hours"] = np.where(speed > 0, left / speed.where(speed > 0, 1), 1000).clip(0, 1000)
    out["has_growth"] = (out["area_growth_abs_0_5h"] > 0).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["event_start_hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["event_start_hour"] / 24)
    return out


class FeatureAdder(BaseEstimator, TransformerMixin):
    """Pipeline step that calls add_features()."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "FeatureAdder":
        """Nothing to learn."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X with the extra features."""
        return add_features(pd.DataFrame(X))
