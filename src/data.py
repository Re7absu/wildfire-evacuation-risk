"""Load the data and build the target labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DIR: Path = ROOT / "data" / "raw"

ID_COL = "event_id"
TIME_COL = "time_to_hit_hours"
EVENT_COL = "event"
TARGET_COLS = [TIME_COL, EVENT_COL]
HORIZONS = (12, 24, 48, 72)  # hours


def load_raw(data_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) from the raw folder."""
    return pd.read_csv(Path(data_dir) / "train.csv"), pd.read_csv(Path(data_dir) / "test.csv")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns except the id and the targets."""
    return [c for c in df.columns if c not in (ID_COL, *TARGET_COLS)]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate fires and turn infinite values into NaN."""
    out = df.drop_duplicates(subset=[ID_COL]).replace([np.inf, -np.inf], np.nan)
    return out.reset_index(drop=True)


def make_labels(time: np.ndarray | pd.Series, event: np.ndarray | pd.Series,
                horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """One label column per horizon, using the competition rule.

    1 = hit by the horizon, 0 = not hit, NaN = unknown (we stopped watching the
    fire before the horizon), and unknown rows are skipped.
    """
    t, e = np.asarray(time, dtype=float), np.asarray(event, dtype=int)
    out = {}
    for h in horizons:
        lab = np.where((e == 1) & (t <= h), 1.0, 0.0)
        lab[(e == 0) & (t <= h)] = np.nan
        out[f"hit_{h}h"] = lab
    return pd.DataFrame(out)
