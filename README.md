# Zeolite CO2 TSA Screening - ML Pipeline

Machine learning pipeline for predicting CO2 capture performance of IZA zeolite frameworks and identifying Pareto-optimal candidates for Temperature Swing Adsorption (TSA) from post-combustion flue gas.

**Target Metrics to Co-Optimize:**
- Maximize CO2 capacity (q_CO2) in mmol/g
- Maximize CO2/N2 selectivity (S_CO2/N2)
- Minimize zeolite regeneration energy (E_R), or CO2 desorption energy, in kJ/mol

## 1.0 Prerequisites
- Python **3.10** (required — `zefram` is incompatible with 3.11+)
- pip
- VS Code with the **Jupyter** and **Python** extensions

To check your Python version: `python --version`
If needed, download Python 3.10 from [python.org](https://www.python.org/downloads/).

## 2.0 Environment Setup
From the project root, create a virtual environment and install all dependencies:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Register the environment as a Jupyter kernel so VS Code can find it:

```bash
pip install ipykernel
python -m ipykernel install --user --name zeolite-ml --display-name "Zeolite ML (3.10)"
```

> **Note on `zefram`:** Notebook 01 uses `zefram` to extract zeolite structural features. Because `zefram` requires Python ≤ 3.10, Notebook 01 automatically creates its own isolated Python 3.10 venv using `uv` (included in `requirements.txt`) and installs `zefram` there. You do not need to install `zefram` manually — just ensure `uv` is available (see `requirements.txt`).

## 3.0 Running the Notebooks

Open the project folder in VS Code. Run the three notebooks in order, since each one produces files to be used by the next.

```
01_data_collection.ipynb
- Output(s): zeolite_features_targets.csv
02_model_training.ipynb
- Output(s): best_models.pkl  model_metadata.json  scaler_nn.pkl
03_pareto_ranking.ipynb
- Output(s): zeolite_rankings.csv
```

### 3.1 01_data_collection.ipynb: Data Collection & Feature Pre-Processing
Extracts 37 structural descriptors (porosity, geometry, topology, lattice parameters) for 229 IZA zeolite frameworks using `zefram`, then assigns three CO2 capture target metrics per framework.

**Target assignment uses a tiered approach:**
1. Experimental data (if available)
2. Physics-based estimation
3. Synthetic/ML-estimated values (default, for demonstration)
**Outputs:** `zeolite_raw_features.csv`, `zeolite_features_targets.csv`

### 3.2 02_model_training.ipynb: Model Training, Tuning, & Comparison
Loads `zeolite_features_targets.csv`, splits 80/20 train/test, and trains 15 model architectures:
- **Single-Target:** Ridge, Lasso, ElasticNet, RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost, SVR, Optuna-tuned XGBoost (80 trials per target)
- **Multi-Output:** Native RF, MultiOutputRegressor (XGBoost), RegressorChain (XGBoost)
- **Multi-Task Neural Network:** PyTorch shared encoder (128->64->32) with 3 task heads

Models are evaluated with Repeated 5-Fold CV (3 repeats) using R2, RMSE, and MAE. Best models are retrained on the full training set and saved.

**Outputs:** `best_models.pkl`, `model_metadata.json`, `scaler_nn.pkl`

### 3.3 03_pareto_ranking.ipynb: Pareto Analysis & Zeolite Ranking
Generates predictions for all 229 zeolite frameworks, identifies the Pareto-optimal set (58 frameworks), and ranks all zeolites by a composite score.

**Composite Score Weights (adjustable in Section 3.3):**
- 40% normalized q_CO2 + 40% normalized S_CO2/N2 + 20% (1 − normalized E_R)
**Outputs:** `zeolite_rankings.csv`

## 4.0 All Output Files
All output files are created at runtime in the project root:

| File | Created by | Used by |
|------|-----------|---------|
| `zeolite_raw_features.csv` | Notebook 01 | Notebook 01 (cache) |
| `zeolite_features_targets.csv` | Notebook 01 | Notebooks 02, 03 |
| `best_models.pkl` | Notebook 02 | Notebook 03 |
| `model_metadata.json` | Notebook 02 | Notebook 03 |
| `scaler_nn.pkl` | Notebook 02 | Notebook 03 |
| `zeolite_rankings.csv` | Notebook 03 | Final output |

## 5.0 Troubleshooting
**`uv` not found**
`pip install uv` (or reinstall `requirements.txt` after activating your venv)

**`zefram` import error**
Notebook 01 manages this automatically. If it fails, ensure `uv` is installed and you are on Python 3.10.

**`ModuleNotFoundError` for any package**
Confirm your venv is activated and run `pip install -r requirements.txt` again.

**Optuna tuning is slow**
In Notebook 02 Section 3.4, reduce `n_trials=80` to `n_trials=30`.

**`paretoset` not found**
`pip install paretoset`. Notebook 03 includes a manual Pareto fallback if the package is missing.