"""Extra features added before training."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Create new columns from existing ones to help the model (mainly distance-based).
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few simple distance and time features."""
    out = df.copy()
    dist = out["dist_min_ci_0_5h"].clip(lower=0)            # metres to the evacuation zone
    left = (dist - 5000).clip(lower=0)                      # metres left before the 5 km line
    speed = out["closing_speed_m_per_h"]
    # Distance on a log scale, because it ranges from metres to hundreds of km.
    out["log_dist_min"] = np.log1p(dist)
    # How far the fire still is from the 5 km "hit" line, in km.
    out["dist_left_km"] = left / 1000
    # Flag: is the fire already within 10 km?
    out["within_10km"] = (dist <= 10_000).astype(int)
    # Rough hours until the fire reaches the 5 km line at its current speed (1000 = not approaching).
    out["eta_hours"] = np.where(speed > 0, left / speed.where(speed > 0, 1), 1000).clip(0, 1000)
    # Flag: did the fire grow at all in the first 5 hours?
    out["has_growth"] = (out["area_growth_abs_0_5h"] > 0).astype(int)
    # Hour of day as a circle, so 23:00 and 00:00 count as close together.
    out["hour_sin"] = np.sin(2 * np.pi * out["event_start_hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["event_start_hour"] / 24)
    return out


# Wrap add_features() as a pipeline step, so it runs the same way in training and prediction.
class FeatureAdder(BaseEstimator, TransformerMixin):
    """Pipeline step that calls add_features()."""

    # Nothing to learn from the data, so fit just returns itself.
    def fit(self, X: pd.DataFrame, y: object = None) -> "FeatureAdder":
        """Nothing to learn."""
        return self

    # Add the extra columns to the data passing through the pipeline.
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X with the extra features."""
        return add_features(pd.DataFrame(X))
