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

# Let this script import the project's own code (src/...) when run from the command line.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import ID_COL  # noqa: E402
from src.model import MAIN_HORIZON, risk_band  # noqa: E402

# Where notebook 02 saved the trained model.
MODEL_PATH = ROOT / "models" / "fire_risk_model.joblib"


# Add risk columns to a table of fires using the saved model.
def predict(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Return df plus risk % per horizon, risk_percentage (48 h) and risk_band."""
    # Stop early with a clear message if any required column is missing.
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")
    out = df.reset_index(drop=True)
    # Use only the model's input columns, with infinite values treated as missing.
    X = out[bundle["features"]].replace([float("inf"), float("-inf")], float("nan"))
    risk = bundle["model"].predict_risk(X).round(2)
    out = pd.concat([out, risk], axis=1)
    # Main risk shown to users = the 48 h risk, plus its Low/Medium/High/Critical band.
    out["risk_percentage"] = out[f"risk_{MAIN_HORIZON}h_pct"]
    out["risk_band"] = risk_band(out["risk_percentage"]).astype(str)
    return out


# Command-line entry point: read options, predict, and save the results.
def main() -> None:
    """Read the input CSV, predict, write the output CSV(s)."""
    # Command-line options.
    parser = argparse.ArgumentParser(description="Predict wildfire risk (%).")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--submission", type=Path, help="optional competition-format file")
    args = parser.parse_args()

    # Load the model and predict.
    bundle = joblib.load(MODEL_PATH)
    try:
        pred = predict(pd.read_csv(args.input), bundle)
    except ValueError as err:
        parser.error(str(err))
    pred.to_csv(args.output, index=False)
    print(f"Saved {len(pred)} predictions to {args.output}")

    # Optionally write the competition file: event_id + probability (0 to 1) per time limit.
    if args.submission:
        sub = pred[[ID_COL]].copy()
        for h in (12, 24, 48, 72):
            sub[f"prob_{h}h"] = pred[f"risk_{h}h_pct"] / 100
        sub.to_csv(args.submission, index=False)
        print(f"Saved submission to {args.submission}")

    # Show a short summary in the terminal.
    print(pred[[c for c in (ID_COL, "risk_percentage", "risk_band") if c in pred]].to_string(index=False))


# Run main() only when this file is run directly (not when imported).
if __name__ == "__main__":
    main()
