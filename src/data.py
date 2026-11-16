"""Data loading, validation and label construction for the fire-risk project.

The raw data comes from the WiDS Global Datathon 2026 wildfire task: one row per
fire event, features describe the first 5 hours of the fire, and the targets say
whether / when the fire came within 5 km of an evacuation zone in the next 72 h.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
TEST_DIR: Path = PROJECT_ROOT / "data" / "test"
MODELS_DIR: Path = PROJECT_ROOT / "models"
FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"

ID_COL: str = "event_id"
TIME_COL: str = "time_to_hit_hours"
EVENT_COL: str = "event"
TARGET_COLS: list[str] = [TIME_COL, EVENT_COL]
HORIZONS: tuple[int, ...] = (12, 24, 48, 72)


def load_raw(data_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the labelled training file and the unlabelled competition test file.

    Args:
        data_dir: Folder holding ``train.csv`` and ``test.csv``.

    Returns:
        ``(train, test)`` data frames exactly as stored on disk.
    """
    train = pd.read_csv(Path(data_dir) / "train.csv")
    test = pd.read_csv(Path(data_dir) / "test.csv")
    return train, test


def load_metadata(data_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load the column dictionary (``metaData.csv``).

    Args:
        data_dir: Folder holding ``metaData.csv``.

    Returns:
        Data frame with one row per column: type, category, description, units.
    """
    return pd.read_csv(Path(data_dir) / "metaData.csv")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model input columns (everything except the id and targets).

    Args:
        df: Raw training or test frame.

    Returns:
        Ordered list of feature column names.
    """
    return [c for c in df.columns if c not in (ID_COL, *TARGET_COLS)]


def validate(df: pd.DataFrame, require_targets: bool = True) -> list[str]:
    """Run basic data-quality checks and return a list of human-readable issues.

    Checks: duplicated ids, duplicated rows, missing values, target ranges.

    Args:
        df: Frame to check.
        require_targets: Whether the target columns must be present and valid.

    Returns:
        List of issue descriptions (empty when the frame looks clean).
    """
    issues: list[str] = []
    if df[ID_COL].duplicated().any():
        issues.append(f"{df[ID_COL].duplicated().sum()} duplicated {ID_COL} values")
    if df.duplicated().any():
        issues.append(f"{df.duplicated().sum()} fully duplicated rows")
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        issues.append(f"{n_missing} missing values")
    if require_targets:
        missing_targets = [c for c in TARGET_COLS if c not in df.columns]
        if missing_targets:
            issues.append(f"missing target columns: {missing_targets}")
        else:
            if not df[EVENT_COL].isin([0, 1]).all():
                issues.append("event column is not binary")
            if (df[TIME_COL] < 0).any() or (df[TIME_COL] > 72).any():
                issues.append("time_to_hit_hours outside [0, 72]")
    return issues


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy: duplicates dropped, inf replaced by NaN.

    Imputation is intentionally *not* done here; it lives inside the sklearn
    pipeline so it is fitted on training folds only (no leakage).

    Args:
        df: Raw frame.

    Returns:
        Cleaned copy of ``df``.
    """
    out = df.drop_duplicates(subset=[ID_COL]).copy()
    num = out.select_dtypes(include="number").columns
    out[num] = out[num].replace([np.inf, -np.inf], np.nan)
    return out.reset_index(drop=True)


def horizon_labels(
    time: np.ndarray | pd.Series,
    event: np.ndarray | pd.Series,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """Build one binary label per horizon: did the fire hit within ``h`` hours?

    Per the data dictionary, ``event == 0`` means the fire never came within 5 km
    of an evacuation zone during the 72 h window, so censored rows are true
    negatives at every horizon and the label is fully defined.

    Args:
        time: ``time_to_hit_hours`` values.
        event: ``event`` indicator values (1 = hit, 0 = never hit).
        horizons: Horizons in hours.

    Returns:
        Data frame with columns ``hit_<h>h`` (0/1 integers).
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    return pd.DataFrame(
        {f"hit_{h}h": ((e == 1) & (t <= h)).astype(int) for h in horizons}
    )
