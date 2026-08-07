# ML_EDA_full_fledge: Loan Default EDA & Feature Selection Project

A production-grade, end-to-end Exploratory Data Analysis (EDA), Feature Engineering, and Feature Selection project structured according to the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) (CCDS) standard.

---

## Directory Structure

```text
.
├── .github
│   └── workflows
│       └── ci.yml             <- CI/CD pipeline (GitHub Actions for linting, testing, and pipeline execution)
├── LICENSE                    <- Open-source MIT License
├── Makefile                   <- Automation commands (`make venv`, `make data`, `make features`, `make train`, etc.)
├── README.md                  <- Top-level project documentation
├── pyproject.toml             <- Project packaging, Ruff linter, and Pytest configuration
├── requirements.txt           <- Dependencies for reproducing the environment
│
├── data
│   ├── raw                    <- Original, immutable raw dataset (data/raw/Loan_Default.csv)
│   ├── interim                <- Intermediate data transformed during pipeline execution
│   └── processed              <- Final canonical datasets for modeling (train_processed.csv, test_processed.csv)
│
├── docs                       <- Project documentation (docs/index.md)
│
├── models                     <- Trained and serialized model artifacts (models/baseline_rf_model.joblib)
│
├── notebooks                  <- Jupyter notebooks (notebooks/1.0-eda-and-feature-selection.ipynb)
│
├── references                 <- Data dictionaries and explanatory materials (references/data_dictionary.md)
│
├── reports                    <- Generated analysis reports & metrics
│   ├── metrics.json           <- Quantitative model evaluation metrics (ROC-AUC, PR-AUC, F1, Confusion Matrix)
│   └── figures                <- Generated graphics (roc_curve.png, pr_curve.png, confusion_matrix.png, etc.)
│
├── src                        <- Source code for execution in this project
│   ├── __init__.py            <- Makes src a Python package
│   ├── config.py              <- Centralized paths and feature drop/keep configuration
│   ├── dataset.py             <- Data loading and dataset verification routines
│   ├── features.py            <- Feature engineering & scikit-learn ColumnTransformer pipeline
│   ├── train.py               <- Entry point for model training
│   ├── predict.py             <- Entry point for batch inference
│   ├── reports.py             <- Report & figure generation script
│   ├── api.py                 <- Single application scoring API wrapper
│   ├── plots.py               <- Reusable plotting routines
│   └── modeling
│       ├── __init__.py
│       ├── train.py           <- Model training & serialization engine
│       └── predict.py         <- Inference engine
│
└── tests                      <- Automated unit & integration test suite (Pytest)
    ├── __init__.py
    ├── conftest.py            <- Pytest fixtures
    ├── test_dataset.py        <- Data loading unit tests
    ├── test_features.py       <- Feature engineering & pipeline unit tests
    └── test_modeling.py       <- Model prediction & probability bounds unit tests
```

---

## Quickstart & Environment Setup

### 1. Create Virtual Environment & Install Dependencies

```bash
# Create virtual environment (.venv) and install dependencies
make venv

# Activate virtual environment
source .venv/bin/activate
```

---

## Running the End-to-End Pipeline

You can run individual pipeline stages via `make` or direct module execution:

```bash
# 1. Verify Dataset Loading & Schema
make data

# 2. Feature Engineering & Preprocessing Pipeline
# Processes raw data, creates missingness flags, constructs financial ratios, fits ColumnTransformer, and saves train/test CSVs
make features

# 3. Model Training & Evaluation
# Trains Logistic Regression and Random Forest models, evaluates performance, and saves model binary to models/
make train

# 4. Batch Prediction Inference
# Generates default probabilities on test data
make predict

# 5. Generate Metrics & Evaluation Figures
# Creates reports/metrics.json and saves figures to reports/figures/
make reports

# 6. Sample Single Application API Inference
make api

# 7. Run Automated Pytest Unit Tests
make test

# 8. Lint Codebase with Ruff
make lint

# 9. Execute Full End-to-End Pipeline
make all
```

---

## Key Business & Technical Findings

1. **Target Leakage Remediation:** Post-decision interest rate attributes (`rate_of_interest`, `Interest_rate_spread`, `Upfront_charges`) were identified as future leakage (missing in 100% of defaults) and removed from pre-underwriting decisioning.
2. **Missingness Indicator Flags:** Missing property appraisals (`property_value_isna`) and debt ratios (`dtir1_isna`) carried a **+0.4132** correlation with default, converting missingness into high-precision binary predictors.
3. **Compound Risk Surfaces:** Debt-to-Income multiplied by Loan-to-Value (`DTI_x_LTV`) captured non-linear layered risk.
4. **Benchmark Model Performance:**
   - **ROC-AUC:** **0.8847**
   - **PR-AUC:** **0.8272**
   - **Accuracy:** **87.92%**
