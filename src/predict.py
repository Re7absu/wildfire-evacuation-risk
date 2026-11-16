"""Score new fires with the saved model.

Usage::

    uv run python src/predict.py --input data/test/sample_input.csv --output predictions.csv
    uv run python src/predict.py --input data/raw/test.csv --output predictions.csv \
        --submission reports/submission.csv

The output keeps every input column and adds:

* ``risk_12h_pct`` ... ``risk_72h_pct`` - probability (%) that the fire comes
  within 5 km of an evacuation zone within each horizon;
* ``risk_percentage`` - the headline risk, equal to the 72 h risk;
* ``risk_band`` - Low (0-25), Medium (25-50), High (50-75), Critical (75-100).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ID_COL  # noqa: E402
from src.model import risk_band  # noqa: E402

DEFAULT_MODEL: Path = ROOT / "models" / "fire_risk_model.joblib"


def load_bundle(model_path: Path = DEFAULT_MODEL) -> dict:
    """Load the saved model bundle (model + feature list + metadata).

    Args:
        model_path: Path to the ``.joblib`` file written by notebook 02.

    Returns:
        Dict with keys ``model``, ``feature_columns``, ``horizons`` and metadata.
    """
    return joblib.load(model_path)


def predict_frame(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Add per-horizon risk %, the headline ``risk_percentage`` and ``risk_band``.

    Args:
        df: New data containing at least the model's feature columns.
        bundle: Output of :func:`load_bundle`.

    Returns:
        ``df`` with the prediction columns appended.

    Raises:
        ValueError: If required feature columns are missing.
    """
    features: list[str] = bundle["feature_columns"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing {len(missing)} required column(s): {missing}")
    X = df[features].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    risk = bundle["model"].predict_risk(X).round(2)
    out = df.reset_index(drop=True).copy()
    out[risk.columns] = risk.to_numpy()
    out["risk_percentage"] = risk[f"risk_{max(bundle['horizons'])}h_pct"].to_numpy()
    out["risk_band"] = risk_band(out["risk_percentage"]).astype(str).to_numpy()
    return out


def to_submission(pred: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Convert predictions to the competition format (``event_id, prob_12h, ...``)."""
    sub = pd.DataFrame({ID_COL: pred[ID_COL]})
    for h in horizons:
        sub[f"prob_{h}h"] = pred[f"risk_{h}h_pct"] / 100.0
    return sub


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Predict wildfire evacuation-zone risk (%).")
    parser.add_argument("--input", required=True, type=Path, help="CSV with the feature columns")
    parser.add_argument("--output", required=True, type=Path, help="Where to write predictions CSV")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to saved model bundle")
    parser.add_argument("--submission", type=Path, default=None, help="Optional competition-format CSV")
    args = parser.parse_args(argv)

    bundle = load_bundle(args.model)
    df = pd.read_csv(args.input)
    try:
        pred = predict_frame(df, bundle)
    except ValueError as err:
        parser.error(str(err))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.output, index=False)
    print(f"Wrote {len(pred)} predictions to {args.output}")
    if args.submission is not None:
        if ID_COL not in pred.columns:
            parser.error(f"--submission needs an '{ID_COL}' column in the input")
        args.submission.parent.mkdir(parents=True, exist_ok=True)
        to_submission(pred, list(bundle["horizons"])).to_csv(args.submission, index=False)
        print(f"Wrote competition submission to {args.submission}")
    cols = [c for c in [ID_COL, "risk_percentage", "risk_band"] if c in pred.columns]
    print(pred[cols].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
