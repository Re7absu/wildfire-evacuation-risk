# Wildfire Evacuation Risk Model: Project Guide

How to set up, run and present the project, step by step.

---

## Contents

1. [What this project does](#1-what-this-project-does)
2. [What is in the folder](#2-what-is-in-the-folder)
3. [One-time setup](#3-one-time-setup)
4. [Run the full project](#4-run-the-full-project)
5. [Predict risk for new fires](#5-predict-risk-for-new-fires)
6. [Build the results dashboard](#6-build-the-results-dashboard)
7. [Troubleshooting](#7-troubleshooting)
8. [Presenting the project](#8-presenting-the-project)
9. [Pre-presentation checklist](#9-pre-presentation-checklist)

---

## 1. What this project does

When a wildfire starts, emergency teams must decide quickly which communities to warn. This project uses **survival analysis** to predict **when a fire will get within 5 km of an evacuation zone**, and the **chance (%) it happens within 12, 24, 48 and 72 hours**, using only the first 5 hours of data about the fire.

| Item | Value |
|---|---|
| Data | WiDS Global Datathon 2026: 221 labelled fires (69% censored), 95 new fires |
| Approach | Survival analysis: Cox Proportional Hazards and Random Survival Forest, compared with a classifier per time limit |
| Final model | **Random Survival Forest** (one model for the whole time-to-hit curve) |
| Score | Hybrid = 0.3 × C-index + 0.7 × (1 − Brier), the competition's metric |
| Result | **0.972** on unseen fires (a simple guess scores 0.738); predicted hit times within **1.6 h** (median) |

---

## 2. What is in the folder

```
fire-risk-model/
├── data/raw/                 original competition files (never changed)
├── data/processed/           cleaned training data          ← made by notebook 01
├── data/test/                test split + sample input/output ← made by notebooks 02 and 03
├── notebooks/
│   ├── 01_eda.ipynb          explore the data, incl. the Kaplan-Meier curve (4 charts)
│   ├── 02_training.ipynb     compare survival and classification models, tune, save the best
│   └── 03_evaluation.ipynb   test on unseen fires: scores, survival curves, time to hit
├── src/
│   ├── data.py               load data, build labels
│   ├── features.py           extra features (distance, timing)
│   ├── survival.py           survival models (Cox PH, RSF) + Kaplan-Meier curve
│   ├── model.py              pipeline, per-time-limit classifier (comparison) + competition score
│   ├── predict.py            command line: predict a CSV file
│   ├── export_results.py     export numbers for the dashboard
│   └── build_dashboard_html.py  build reports/dashboard.html
├── models/fire_risk_model.joblib   the saved model ← made by notebook 02
├── reports/
│   ├── figures/              all charts as PNG
│   ├── dashboard.html        results dashboard (open in a browser)
│   └── submission.csv        competition submission
├── docs/PROJECT_GUIDE.md     this guide
├── README.md                 project summary
└── pyproject.toml, uv.lock   exact package versions
```

---

## 3. One-time setup

The project runs in **WSL (Ubuntu on Windows)** and uses **uv** to manage Python and packages.

### 3.1 Install WSL (skip if you already have it)

Open **PowerShell as Administrator** and run:
```bash
wsl --install -d Ubuntu
```
Restart the computer, open **Ubuntu** from the Start menu, and create a username and password.

### 3.2 Install uv inside WSL

In the Ubuntu terminal:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Close and reopen the terminal, then check that it worked:
```bash
uv --version
```

### 3.3 Go to the project folder

Windows drives appear under `/mnt/c/` in WSL. Use quotes, because the path contains a space:
```bash
cd "/mnt/c/Users/<your-windows-user>/OneDrive/Desktop/New folder/fire-risk-model"
```

### 3.4 Keep the Python environment outside OneDrive

The project is inside OneDrive, and OneDrive would try to sync thousands of package files. Tell uv to keep the environment in your Linux home folder instead. Run this **once**, and it stays set in every new terminal:
```bash
echo 'export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/fire-risk-model' >> ~/.bashrc && source ~/.bashrc
```

### 3.5 Install the packages

```bash
uv sync
```
This installs **Python 3.12** and the exact package versions in `uv.lock` (pandas, numpy, scikit-learn, **scikit-survival**, matplotlib, jupyterlab and others). The project uses Python 3.12 because scikit-survival, which provides Cox PH and Random Survival Forest, is not yet available for newer versions.

### 3.6 Register the Jupyter kernel

```bash
uv run python -m ipykernel install --user --name fire-risk --display-name "Fire Risk (uv)"
```

Setup is done. You only repeat these steps on a new computer.

---

## 4. Run the full project

Always start from the project folder in WSL (step 3.3).

### Option A: run everything with one command (about 1 minute)

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb notebooks/02_training.ipynb notebooks/03_evaluation.ipynb
```

### Option B: run the notebooks one by one (best for learning or demos)

```bash
uv run jupyter lab --no-browser
```
Ctrl+click the `http://localhost:8888/...` link it prints. Then open each notebook and choose **Run → Run All Cells**, in this order:

| Order | Notebook | What it does | Creates |
|---|---|---|---|
| 1 | `01_eda.ipynb` | Explores the data, draws the Kaplan–Meier curve | `data/processed/train_clean.csv`, 4 charts |
| 2 | `02_training.ipynb` | Compares Cox PH, Random Survival Forest, a classifier and a baseline; tunes them; saves the best survival model | `models/fire_risk_model.joblib`, `data/test/test_set.csv` |
| 3 | `03_evaluation.ipynb` | Scores unseen fires, shows survival curves and predicted hit times, predicts new fires | `reports/submission.csv`, 4 charts |

Press **Ctrl+C** in the terminal to stop Jupyter when you are done.

**In VS Code instead:** install the **WSL**, **Python** and **Jupyter** extensions. Press **F1**, run **WSL: Reopen Folder in WSL**, open a notebook, and pick the **Fire Risk (uv)** kernel.

### What a good run looks like

- No red error cells in any notebook.
- Notebook 02 prints `Final model: Random Survival Forest (tuned) | CV Hybrid: 0.978`.
- Notebook 03 shows a test Hybrid score of about **0.972** and a median timing error of about **1.6 hours**.

---

## 5. Predict risk for new fires

The input must be a CSV with the same 34 feature columns as `data/raw/test.csv`.

```bash
uv run python src/predict.py --input data/test/sample_input.csv --output predictions.csv
```

The output keeps your columns and adds:

| Column | Meaning |
|---|---|
| `predicted_hours_to_hit` | When the fire is expected to reach the zone (the hour its risk passes 50%); empty = not expected to hit |
| `risk_12h_pct` … `risk_72h_pct` | Chance (%) of reaching an evacuation zone within each time limit |
| `risk_percentage` | The main risk: the **48 h** risk |
| `risk_band` | Low (0–25%), Medium (25–50%), High (50–75%), Critical (75–100%) |

To also write the competition submission file:
```bash
uv run python src/predict.py --input data/raw/test.csv --output reports/test_predictions.csv --submission reports/submission.csv
```

---

## 6. Build the results dashboard

The dashboard is one HTML file for stakeholders: a scorecard, 6 charts (including the Kaplan–Meier survival curve), and a table of every new fire with its predicted hit time, its risk, and a band filter.

```bash
uv run python src/export_results.py
```
```bash
uv run python src/build_dashboard_html.py
```

Then open `reports/dashboard.html` by double-clicking it in Windows Explorer. It needs internet the first time, to load the chart library.

Re-run both commands whenever you retrain the model, so the dashboard shows the new numbers.

---

## 7. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `uv: command not found` | uv not installed, or the terminal was not reopened | Redo step 3.2, then open a new terminal |
| A `.venv` folder appears inside the project | `UV_PROJECT_ENVIRONMENT` is not set | Redo step 3.4, delete the `.venv` folder, then run `uv sync` |
| `No module named 'sksurv'` | Packages not installed, or the wrong kernel is selected | Run `uv sync`, then pick the **Fire Risk (uv)** kernel |
| scikit-survival fails to install | A Python other than 3.12 is being used | Keep `.python-version` at `3.12` and run `uv sync` again |
| `No such file or directory` with the path | The space in "New folder" | Put the path in double quotes |
| `FileNotFoundError: train_clean.csv` | Notebooks run out of order | Run 01, then 02, then 03 |
| `FileNotFoundError: fire_risk_model.joblib` | The model has not been trained yet | Run notebook 02 |
| `Input is missing columns: [...]` | Wrong CSV given to `predict.py` | Use a file with the same columns as `data/raw/test.csv` |
| Kernel "Fire Risk (uv)" not listed | Kernel not registered | Redo step 3.6, then reload VS Code or Jupyter |
| Dashboard opens but charts are empty | No internet, so d3 could not load | Connect once and refresh the page |
| Jupyter link does not open | WSL port forwarding | Copy the full `http://localhost:8888/lab?token=...` link into Chrome or Edge |

---

## 8. Presenting the project

### 8.1 Know your audience

| Audience | Focus on | Avoid |
|---|---|---|
| Emergency managers, decision makers | When a fire is expected to hit, the 48 h risk, the risk bands, limits | Metric formulas, model settings |
| Data or technical team | Censoring, Kaplan–Meier, Cox PH vs. Random Survival Forest, calibration, scores | Over-explaining basics |
| Judges or instructors | Why survival analysis, honest evaluation, what you would do next | Claiming it is ready for real deployment |

### 8.2 Suggested 10-minute talk

| Time | Section | Show | Key message |
|---|---|---|---|
| 0:00–1:00 | The problem | Title / dashboard header | Teams must decide whom to warn before they are certain. We predict *when* a fire reaches a zone, and the % risk by 12, 24, 48 and 72 hours. |
| 1:00–2:30 | Why survival analysis | `01_target.png` + `01_kaplan_meier.png` | 69% of fires were never seen to hit: they are censored. Survival analysis uses them properly. The Kaplan–Meier curve shows most hits come in the first 12 hours. |
| 2:30–4:00 | Key finding | `01_distance.png` or the dashboard's distance strip | Every fire that hit started within 4.7 km of a zone, and every other fire was more than 5.2 km away. *Whether* is easy; *when* is hard. |
| 4:00–5:30 | The models | `02_model_comparison.png` | Compared Cox PH and Random Survival Forest with a classifier and a baseline. The Random Survival Forest is best: 0.978, against 0.738 for a simple guess. |
| 5:30–7:00 | Results | `03_survival_curves.png`, `03_time_to_hit.png`, dashboard scorecard | On unseen fires: Hybrid 0.972. Predicted hit times are within 1.6 hours (median), and fires that never hit get no hit time. |
| 7:00–8:30 | Live demo | Dashboard table filtered to Critical, or `predict.py` | 28 of 95 new fires are Critical at 48 hours. Show one fire's predicted hit time and its 12 / 24 / 48 h risk. |
| 8:30–10:00 | Limits and next steps | Dashboard "Limitations" | Small data, relies on distance, a few late hits predicted too early, no data beyond about 67 hours. Next: add weather, wind and terrain. |

### 8.3 Speaking points

**Opening:** "When a wildfire starts, emergency managers have to decide quickly which communities to warn. We built a survival model that predicts *when* a fire will reach an evacuation zone, and the chance it happens within 12, 24, 48 and 72 hours, using only its first 5 hours."

**Why survival analysis:** "Most fires in our data were never seen to reach a zone. We only know they hadn't hit by the time tracking stopped. A normal classifier has to guess what happened to them; a survival model uses exactly what we know."

**Key finding:** "Distance explained almost everything. Every fire that reached a zone started within 4.7 kilometres of it. So predicting *whether* a fire will hit is easy. The real challenge is predicting *when*."

**Results:** "Our Random Survival Forest scored 0.972 on fires it had never seen, compared with 0.738 for a simple guess. For fires that did hit, its predicted time was within about an hour and a half (median). And its percentages are honest: when it says 80%, about 80% of those fires really do hit."

**Close:** "The model is a strong first step for triage. With weather and wind data, we expect it to get better at the hardest part: timing."

### 8.4 Live demo script (2 minutes)

1. Open `reports/dashboard.html` before the talk and keep it in a browser tab.
2. Point to the **scorecard**: 0.972, 94% ranking accuracy, 28 Critical fires.
3. Show the **Kaplan–Meier curve**: "the red line, fires that started within 5 km, rises almost to 100%; the grey line, fires beyond 5 km, stays at zero."
4. In **Risk for every new fire**, set **Band** to **Critical** and point to the top row: "this fire is 0.1 km away. The model expects it to hit in about half an hour, with a 98% risk within 12 hours."
5. Optional, in the terminal: run the `predict.py` command from section 5 and show the `predicted_hours_to_hit` column in `predictions.csv`.

Have screenshots of the dashboard ready as a backup in case the internet or projector fails.

### 8.5 Questions you may get

| Question | Suggested answer |
|---|---|
| Why survival analysis instead of classification? | 69% of fires are censored. Survival models use the time each fire was watched, predict *when* a fire hits, and give one consistent risk curve. In our comparison the Random Survival Forest also scored highest (0.978 vs. 0.976 for the best classifier). |
| What is censoring? | A fire we stopped watching before it reached a zone. We know it hadn't hit *up to that time*, but not after. Survival models use that partial information instead of throwing it away. |
| Why did Cox PH score lower? | Cox PH assumes each feature has a fixed, smooth effect on the hit rate. This data has a sharp rule (within about 5 km or not) that a forest captures much better. |
| How is the predicted hit time worked out? | From each fire's survival curve: it is the hour when the chance of a hit passes 50%. If it never passes 50%, the fire is not expected to hit. |
| Isn't using distance cheating? | No. Distance is measured in the first 5 hours, before the forecast window starts. It is a real early warning signal. |
| How reliable is 0.972? | It comes from 34 held-out fires, so it has some uncertainty. Cross-validation (0.978) agrees with it. |
| Why is the 72 h risk the same as the 67 h risk? | No fire was tracked beyond about 67 hours, so the model cannot see later hits. The curve is held flat after that point. |
| Can this be used in real operations? | Not yet. It needs more data, weather inputs, and testing on other regions and seasons first. |
| What would you improve first? | Add wind, weather and terrain; validate on more fires; try more survival models (for example gradient-boosted survival models). |

### 8.6 Presentation tips

- **Lead with the finding, not the method.** "Distance decides, and we predict *when*" is more memorable than "we used a Random Survival Forest".
- **Explain censoring in one sentence** the first time: "fires we stopped watching before they hit".
- **Say each number once and clearly**, and round it when speaking: "0.97" or "about an hour and a half".
- **Explain each metric in one plain sentence** the first time you use it.
- **Be upfront about limitations.** It builds trust.
- **Use the predicted hit time and the 48 h risk** as the main numbers for non-technical audiences.
- **Don't show code** to decision makers. Show the dashboard.

---

## 9. Pre-presentation checklist

- [ ] All three notebooks run with no errors (section 4, option A).
- [ ] `reports/dashboard.html` is rebuilt and opens with charts (section 6).
- [ ] Dashboard tab is open and the browser is zoomed so it fits the screen.
- [ ] Backup screenshots of the dashboard and key charts in `reports/figures/`.
- [ ] `predict.py` demo command tested in the terminal (if you plan to show it).
- [ ] Key numbers memorised: **221 fires · 69% censored · 4.7 / 5.2 km · RSF 0.978 CV / 0.972 test · 0.738 baseline · 1.6 h median timing error · 28 of 95 Critical**.
- [ ] Answers to the questions in 8.5 rehearsed once out loud.
- [ ] Talk timed at 10 minutes or less.
