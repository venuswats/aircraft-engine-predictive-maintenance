# Aircraft Engine Predictive Maintenance — Remaining Useful Life (RUL)

**Academic project | Computer Science | Python, TensorFlow/Keras, scikit-learn**

This project investigates regression-based estimation of Remaining Useful Life (RUL) for simulated turbofan engines using sensor measurements and operating settings from a C-MAPSS-style public dataset. It is an academic experiment, not a deployed aviation maintenance system.

## Objective

Estimate the number of operating cycles remaining before an engine reaches its end of life, using historical sensor and operating-condition features.

## Workflow

1. **Preprocessing** — validate numeric columns, construct training RUL labels, and cap training targets at 130 cycles.
2. **Exploratory analysis** — inspect feature distributions and correlations.
3. **Training** — train a multilayer perceptron (MLP) with 128 and 64 hidden units, ReLU activations, 20% dropout, Adam optimizer and mean squared error loss.
4. **Validation** — hold out complete engine units using `GroupShuffleSplit`, so cycles from the same engine do not appear in both training and validation subsets.
5. **Scaling** — fit `MinMaxScaler` only on the training subset and reuse the saved scaler for validation/test transformation.
6. **Evaluation** — calculate RMSE, MAE and R² on the final observed cycle of each test engine, using the supplied ground-truth RUL labels.

## Repository structure

```text
.
├── data/
│   └── raw/                 # Place the three source CSV files here (not committed)
├── outputs/
│   ├── figures/             # Generated charts (not committed)
│   ├── models/              # Trained model, scaler and metadata (not committed)
│   └── tables/              # Generated tables and metrics (not committed)
├── src/
│   ├── utils.py
│   ├── 01_preprocess.py
│   ├── 02_eda.py
│   ├── 03_train_model.py
│   └── 04_evaluate_model.py
├── REQUIREMENTS.txt
└── README.md
```

## Dataset setup

Place the original files in `data/raw/` with these exact names:

- `PM_train.csv`
- `PM_test.csv`
- `PM_truth.csv`

The scripts expect headerless comma-separated files in the conventional C-MAPSS-derived layout: unit, cycle, three operating settings, then sensor columns. The dataset is not included in this repository. Obtain it from its original/public source and check its terms before sharing or redistributing it. Add a dataset citation and exact variant (for example, FD001) here after confirming the source used in the final experiment.

## Setup

Python 3.10–3.12 is recommended. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r REQUIREMENTS.txt
```

## Run the pipeline

Run commands from the repository root:

```bash
python src/01_preprocess.py
python src/02_eda.py
python src/03_train_model.py
python src/04_evaluate_model.py
```

The training script saves the best validation checkpoint, the fitted scaler, and feature metadata under `outputs/models/`. The evaluation script loads those exact artifacts, preventing feature-order and scaling mismatches.

## Evaluation protocol and interpretation

- The validation split is grouped by engine (`unit`) to reduce leakage between correlated cycles from the same engine.
- The scaler is fitted only on training engines.
- The test metrics use one row per test engine: its last observed cycle. The corresponding true RUL comes from `PM_truth.csv`.
- Training targets are capped at 130 cycles; test ground-truth labels are kept as supplied for evaluation. This difference should be considered when interpreting errors for engines with high remaining life.
- Results must be regenerated from the exact dataset variant and software environment before being reported in a thesis or presentation. Do not copy old metrics into this README.

## Reproducibility and limitations

A fixed random seed is used, but exact numerical reproducibility can vary by TensorFlow version, hardware and execution backend. The project uses simulated/public data and does not establish operational readiness, airworthiness, or suitability for real maintenance decisions. Further work could compare alternative regressors, tune hyperparameters, and evaluate additional C-MAPSS subsets.

## Author

**Sophia Rocha**  
Computer Science | Data Science & Machine Learning

> Before publishing: verify the dataset source/license, remove any personal or confidential material, and only commit generated model artifacts if their size and redistribution terms are appropriate.
