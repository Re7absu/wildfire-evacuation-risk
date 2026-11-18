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

When a wildfire starts, emergency teams must decide quickly which communities to warn. This project predicts the **chance (%) that a fire gets within 5 km of an evacuation zone** within **12, 24, 48 and 72 hours**, using only the first 5 hours of data about the fire.

| Item | Value |
|---|---|
| Data | WiDS Global Datathon 2026: 221 labelled fires, 95 new fires |
| Model | Random Forest, one per time limit (12 / 24 / 48 / 72 h) |
| Score | Hybrid = 0.3 × C-index + 0.7 × (1 − Brier), the competition's metric |
| Result | **0.971** on unseen fires (a simple guess scores 0.738) |

---

## 2. What is in the folder

```
fire-risk-model/
├── data/raw/                 original competition files (never changed)
├── data/processed/           cleaned training data          ← made by notebook 01
├── data/test/                test split + sample input/output ← made by notebooks 02 and 03
├── notebooks/
│   ├── 01_eda.ipynb          explore the data (3 charts)
│   ├── 02_training.ipynb     compare models, tune, save the best
│   └── 03_evaluation.ipynb   test on unseen fires, predict new ones
├── src/
│   ├── data.py               load data, build labels
│   ├── features.py           extra features (distance, timing)
│   ├── model.py              the model + competition score
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
This installs the exact versions in `uv.lock` (pandas, numpy, scikit-learn, matplotlib, jupyterlab and others).

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
| 1 | `01_eda.ipynb` | Explores the data | `data/processed/train_clean.csv`, 3 charts |
| 2 | `02_training.ipynb` | Compares 4 models, tunes the best | `models/fire_risk_model.joblib`, `data/test/test_set.csv` |
| 3 | `03_evaluation.ipynb` | Scores unseen fires, predicts new ones | `reports/submission.csv`, 4 charts |

Press **Ctrl+C** in the terminal to stop Jupyter when you are done.

**In VS Code instead:** install the **WSL**, **Python** and **Jupyter** extensions. Press **F1**, run **WSL: Reopen Folder in WSL**, open a notebook, and pick the **Fire Risk (uv)** kernel.

### What a good run looks like

- No red error cells in any notebook.
- Notebook 02 prints a tuned Hybrid score of about **0.976**.
- Notebook 03 shows a test Hybrid score of about **0.971**.

---

## 5. Predict risk for new fires

The input must be a CSV with the same 34 feature columns as `data/raw/test.csv`.

```bash
uv run python src/predict.py --input data/test/sample_input.csv --output predictions.csv
```

The output keeps your columns and adds:

| Column | Meaning |
|---|---|
| `risk_12h_pct` … `risk_72h_pct` | Chance (%) of reaching an evacuation zone within each time limit |
| `risk_percentage` | The main risk: the **48 h** risk |
| `risk_band` | Low (0–25%), Medium (25–50%), High (50–75%), Critical (75–100%) |

To also write the competition submission file:
```bash
uv run python src/predict.py --input data/raw/test.csv --output reports/test_predictions.csv --submission reports/submission.csv
```

---

## 6. Build the results dashboard

The dashboard is one HTML file for stakeholders: a scorecard, 6 charts, and a table of every new fire with a band filter.

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
| Emergency managers, decision makers | What the risk % means, the 48 h forecast, the risk bands, limits | Metric formulas, model settings |
| Data or technical team | Labels and censoring, model comparison, calibration, scores | Over-explaining basics |
| Judges or instructors | Method, honest evaluation, what you would do next | Claiming it is ready for real deployment |

### 8.2 Suggested 10-minute talk

| Time | Section | Show | Key message |
|---|---|---|---|
| 0:00–1:00 | The problem | Title / dashboard header | Teams must decide whom to warn before they are certain. We give a % risk for 12, 24, 48 and 72 hours. |
| 1:00–2:30 | The data | `01_target.png` | 221 fires, and only 31% reached an evacuation zone. Most of those did so within 12 hours (49 of 69). |
| 2:30–4:00 | Key finding | `01_distance.png` or the dashboard's distance strip | Every fire that hit started within 4.7 km of a zone, and every other fire was more than 5.2 km away. *Whether* is easy; *when* is hard. |
| 4:00–5:30 | The model | `02_model_comparison.png` | Compared 4 models with 5-fold cross-validation. The tuned Random Forest is best: 0.976, against 0.738 for a simple guess. |
| 5:30–7:00 | Results | Dashboard scorecard + calibration chart | On unseen fires: Hybrid 0.971, ranking accuracy 93%. The percentages are trustworthy (the points sit on the diagonal). |
| 7:00–8:30 | Live demo | Dashboard table filtered to Critical, or `predict.py` | 26 of 95 new fires are Critical at 48 hours. Show one fire's 12 / 24 / 48 h risk. |
| 8:30–10:00 | Limits and next steps | Dashboard "Limitations" | Small data, relies on distance, the 72 h value is a scoring side effect. Next: add weather, wind and terrain. |

### 8.3 Speaking points

**Opening:** "When a wildfire starts, emergency managers have to decide quickly which communities to warn. We built a model that estimates the chance a fire reaches an evacuation zone within 12, 24, 48 and 72 hours, using only its first 5 hours."

**Key finding:** "Distance explained almost everything. Every fire that reached a zone started within 4.7 kilometres of it. So predicting *whether* a fire will hit is easy. The real challenge is predicting *when*."

**Results:** "On fires the model had never seen, it scored 0.971 on the competition's Hybrid score, compared with 0.738 for a simple guess. More importantly, its percentages are honest: when it says 80%, about 80% of those fires really do hit."

**Close:** "The model is a strong first step for triage. With weather and wind data, we expect it to get better at the hardest part: timing."

### 8.4 Live demo script (2 minutes)

1. Open `reports/dashboard.html` before the talk and keep it in a browser tab.
2. Point to the **scorecard**: 0.971, 93%, 26 Critical fires.
3. Scroll to the **distance strip**: "red dots are fires that hit, and they all sit left of the 5 km line."
4. In **Risk for every new fire**, set **Band** to **Critical** and hover over the top row: "this fire is 1.2 km away and has a 99% risk within 12 hours."
5. Optional, in the terminal: run the `predict.py` command from section 5 and show `predictions.csv`.

Have screenshots of the dashboard ready as a backup in case the internet or projector fails.

### 8.5 Questions you may get

| Question | Suggested answer |
|---|---|
| Why is the 72 h risk about 98% for every fire? | Under the competition's scoring rule, every fire that can be judged at 72 h is a hit, because fires that did not hit were last observed before 72 h. So the best score comes from a high 72 h value. It is not a real forecast; use 12, 24 and 48 h for decisions. |
| Isn't using distance cheating? | No. Distance is measured in the first 5 hours, before the forecast window starts. It is a real early warning signal. |
| How reliable is 0.971? | It comes from 34 held-out fires, so it has a range: about 0.94 to 0.99. Cross-validation (0.976) agrees with it. |
| Why Random Forest? | It scored best in cross-validation and handles the sharp distance threshold and the many zero values in the data well. |
| What does "skipped" mean? | If a fire stopped being tracked before a time limit, we don't know what happened next, so that fire is left out at that limit. That is the competition's rule. |
| Can this be used in real operations? | Not yet. It needs more data, weather inputs, and testing on other regions and seasons first. |
| What would you improve first? | Add wind, weather and terrain; try survival models; validate on more fires. |

### 8.6 Presentation tips

- **Lead with the finding, not the method.** "Distance decides" is more memorable than "we used a Random Forest".
- **Say each number once and clearly**, and round it when speaking: "about 97 out of 100" or "0.97".
- **Explain each metric in one plain sentence** the first time you use it.
- **Be upfront about limitations.** It builds trust.
- **Use the 48 h risk** as the main number for non-technical audiences.
- **Don't show code** to decision makers. Show the dashboard.

---

## 9. Pre-presentation checklist

- [ ] All three notebooks run with no errors (section 4, option A).
- [ ] `reports/dashboard.html` is rebuilt and opens with charts (section 6).
- [ ] Dashboard tab is open and the browser is zoomed so it fits the screen.
- [ ] Backup screenshots of the dashboard and key charts in `reports/figures/`.
- [ ] `predict.py` demo command tested in the terminal (if you plan to show it).
- [ ] Key numbers memorised: **221 fires · 31% hit · 4.7 / 5.2 km · 0.971 test · 0.738 baseline · 93% ranking · 26 of 95 Critical**.
- [ ] Answers to the questions in 8.5 rehearsed once out loud.
- [ ] Talk timed at 10 minutes or less.
