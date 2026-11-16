"""Predict fire risk for a CSV file.

Example:
    uv run python src/predict.py --input data/test/sample_input.csv --output predictions.csv
    (add --submission reports/submission.csv for the competition file)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import ID_COL  # noqa: E402
from src.model import MAIN_HORIZON, risk_band  # noqa: E402

MODEL_PATH = ROOT / "models" / "fire_risk_model.joblib"


def predict(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Return df plus risk % per horizon, risk_percentage (48 h) and risk_band."""
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")
    out = df.reset_index(drop=True)
    X = out[bundle["features"]].replace([float("inf"), float("-inf")], float("nan"))
    risk = bundle["model"].predict_risk(X).round(2)
    out = pd.concat([out, risk], axis=1)
    out["risk_percentage"] = out[f"risk_{MAIN_HORIZON}h_pct"]
    out["risk_band"] = risk_band(out["risk_percentage"]).astype(str)
    return out


def main() -> None:
    """Read the input CSV, predict, write the output CSV(s)."""
    parser = argparse.ArgumentParser(description="Predict wildfire risk (%).")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--submission", type=Path, help="optional competition-format file")
    args = parser.parse_args()

    bundle = joblib.load(MODEL_PATH)
    try:
        pred = predict(pd.read_csv(args.input), bundle)
    except ValueError as err:
        parser.error(str(err))
    pred.to_csv(args.output, index=False)
    print(f"Saved {len(pred)} predictions to {args.output}")

    if args.submission:
        sub = pred[[ID_COL]].copy()
        for h in (12, 24, 48, 72):
            sub[f"prob_{h}h"] = pred[f"risk_{h}h_pct"] / 100
        sub.to_csv(args.submission, index=False)
        print(f"Saved submission to {args.submission}")

    print(pred[[c for c in (ID_COL, "risk_percentage", "risk_band") if c in pred]].to_string(index=False))


if __name__ == "__main__":
    main()
