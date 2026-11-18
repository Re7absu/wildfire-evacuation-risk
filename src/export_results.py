"""Export the key results as small JSON files for the results dashboard.

Run from the project folder (after notebooks 01-03):
    uv run python src/export_results.py
Writes reports/dashboard/*.json, each as {"rows": [...], "meta": [...]}.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

# Let this script import the project's own code (src/...) and hide noisy library warnings.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.data import HORIZONS, TARGET_COLS, TIME_COL, EVENT_COL, make_labels  # noqa: E402
from src.model import hybrid_scorer, risk_band, score  # noqa: E402
from src.survival import hit_curve  # noqa: E402

# Output folder, and this script's own text (saved with each file so readers can see how numbers were made).
OUT = ROOT / "reports" / "dashboard"
CODE = Path(__file__).read_text(encoding="utf-8")


# Write one dashboard data file as JSON: the rows plus where they came from.
def save(name: str, rows: list[dict], made_from: str) -> None:
    """Write one dataset with the code that made it."""
    meta = [{"label": "export_results.py", "value": CODE, "type": "code", "lang": "python"},
            {"label": "Input", "value": made_from}]
    (OUT / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": meta}, indent=1), encoding="utf-8")
    print("wrote", name, len(rows), "rows")


# Compute every number the dashboard shows and save each group as its own file.
def main() -> None:
    """Compute every number the dashboard shows."""
    # Load the cleaned data, the held-out test split and the saved model.
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "processed" / "train_clean.csv")
    test = pd.read_csv(ROOT / "data" / "test" / "test_set.csv")
    bundle = joblib.load(ROOT / "models" / "fire_risk_model.joblib")
    model, features = bundle["model"], bundle["features"]

    # 1. Fires per time limit (all 221 training fires)
    lab = make_labels(df[TIME_COL], df[EVENT_COL])
    save("label_counts", [{"horizon": f"{h} h", "hit": int((lab[f"hit_{h}h"] == 1).sum()),
                           "no_hit": int((lab[f"hit_{h}h"] == 0).sum()), "skipped": int(lab[f"hit_{h}h"].isna().sum())}
                          for h in HORIZONS], "data/processed/train_clean.csv")

    # 2. Distance vs outcome, one row per fire
    save("fires", [{"event_id": str(r.event_id), "distance_km": round(r.dist_min_ci_0_5h / 1000, 3),
                    "hours": round(getattr(r, TIME_COL), 2), "hit": int(getattr(r, EVENT_COL))}
                   for r in df.itertuples()], "data/processed/train_clean.csv")

    # 3. Model comparison (5-fold CV), saved with the model by notebook 02
    rows = [{"model": r["model"], "type": r["type"], "hybrid": r["hybrid"], "c_index": r["c_index"],
             "brier": r["brier"], "final": bool(r["final"])} for r in bundle["cv_table"]]
    save("model_cv", rows, "models/fire_risk_model.joblib (comparison table from notebook 02)")

    # 3b. Kaplan-Meier curves: share of fires that have hit by each hour (all fires, near vs. far)
    near = df["dist_min_ci_0_5h"] <= 5000
    km = []
    for group_name, mask in [("All fires", np.ones(len(df), bool)), ("Within 5 km", near.to_numpy()), ("Beyond 5 km", ~near.to_numpy())]:
        t, p = hit_curve(df.loc[mask, TIME_COL], df.loc[mask, EVENT_COL])
        km += [{"group": group_name, "hours": 0.0, "pct_hit": 0.0}]
        km += [{"group": group_name, "hours": round(float(a), 2), "pct_hit": round(float(b) * 100, 2)} for a, b in zip(t, p)]
    save("km", km, "data/processed/train_clean.csv")

    # 4. Scores on the 34 held-out test fires
    probs = model.predict_proba(test[features])
    s = score(probs, test[TIME_COL], test[EVENT_COL])
    save("test_scores", [{"metric": k, "value": round(float(v), 4)} for k, v in s.items()],
         "data/test/test_set.csv, models/fire_risk_model.joblib")

    # 5. Calibration on the test set (12/24/48 h, 4 bins each)
    tl = make_labels(test[TIME_COL], test[EVENT_COL])
    cal = []
    for k, h in enumerate((12, 24, 48)):
        known = tl[f"hit_{h}h"].notna().to_numpy()
        real, pred = calibration_curve(tl.loc[known, f"hit_{h}h"], probs[known, k], n_bins=4, strategy="quantile")
        cal += [{"horizon": f"{h} h", "predicted_pct": round(p * 100, 1), "real_pct": round(r * 100, 1)} for p, r in zip(pred, real)]
    save("calibration", cal, "data/test/test_set.csv, models/fire_risk_model.joblib")

    # 6. Feature importance (drop in Hybrid score when a feature is shuffled)
    imp = permutation_importance(model, test[features], test[TARGET_COLS].to_numpy(), scoring=hybrid_scorer,
                                 n_repeats=10, random_state=42, n_jobs=-1)
    order = np.argsort(imp.importances_mean)[::-1][:5]
    save("importance", [{"feature": features[i], "importance": round(float(imp.importances_mean[i]), 4)} for i in order],
         "data/test/test_set.csv, models/fire_risk_model.joblib")

    # 7. Predictions for the 95 competition fires (no answers known)
    new = pd.read_csv(ROOT / "data" / "raw" / "test.csv")
    p = model.predict_proba(new[features]) * 100
    band = risk_band(p[:, 2]).astype(str)
    # Predicted hours to hit from the survival model (None = not expected to hit)
    hours = model.predict_time(new[features]) if hasattr(model, "predict_time") else np.full(len(new), np.nan)
    save("new_fires", [{"event_id": str(e), "distance_km": round(d / 1000, 2),
                        "predicted_hours": None if np.isnan(t) else round(float(t), 1),
                        "risk_12h": round(a, 1), "risk_24h": round(b, 1), "risk_48h": round(c, 1), "band": bd}
                       for e, d, t, a, b, c, bd in zip(new["event_id"], new["dist_min_ci_0_5h"], hours, p[:, 0], p[:, 1], p[:, 2], band)],
         "data/raw/test.csv, models/fire_risk_model.joblib")


# Run main() only when this file is run directly (not when imported).
if __name__ == "__main__":
    main()
