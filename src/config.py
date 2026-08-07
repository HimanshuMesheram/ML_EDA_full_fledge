import pathlib

# Project Root Directory
PROJ_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Data Directory Paths
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "Loan_Default.csv"
TRAIN_PROCESSED_PATH = PROCESSED_DATA_DIR / "train_processed.csv"
TEST_PROCESSED_PATH = PROCESSED_DATA_DIR / "test_processed.csv"

# Models Directory Paths
MODELS_DIR = PROJ_ROOT / "models"
MODEL_PATH = MODELS_DIR / "baseline_rf_model.joblib"

# Reports Directory Paths
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Target Column
TARGET_COL = "Status"

# Post-Decision Target Leakage Features to Drop
LEAKAGE_COLS = [
    "rate_of_interest",
    "Interest_rate_spread",
    "Upfront_charges"
]

# Zero / Near-Zero Variance Features to Drop
ZERO_VAR_COLS = [
    "ID",
    "year",
    "construction_type",
    "Secured_by",
    "Security_Type",
    "open_credit"
]

# Unpredictive Features to Drop
UNPREDICTIVE_COLS = [
    "Credit_Score",
    "term",
    "Gender",
    "Region",
    "approv_in_adv",
    "loan_limit",
    "business_or_commercial",
    "credit_type",
    "Credit_Worthiness",
    "interest_only"
]

# Master Drop List
DROP_FEATURES = list(set(LEAKAGE_COLS + ZERO_VAR_COLS + UNPREDICTIVE_COLS))

# Numerical Pipeline Features
NUMERICAL_FEATURES = [
    "property_value",
    "dtir1",
    "LTV",
    "income",
    "loan_amount",
    "LTV_calculated",
    "Payment_to_Income",
    "Loan_to_Income",
    "DTI_x_LTV"
]

# Binary / Missingness Indicator Features
BINARY_FEATURES = [
    "property_value_isna",
    "dtir1_isna",
    "income_isna",
    "LTV_isna",
    "age_ordinal",
    "total_units_ordinal"
]

# Categorical Features for One-Hot Encoding
CATEGORICAL_FEATURES = [
    "loan_type",
    "loan_purpose",
    "occupancy_type",
    "submission_of_application",
    "Neg_ammortization",
    "lump_sum_payment",
    "co-applicant_credit_type"
]
# # Model Configuration
# DEFAULT_THRESHOLD = 0.9
# DEFAULT_MODEL = "XGBOOOST"
# DEFAULT_MODEL_PATH = MODELS_DIR / "baseline_rf_model.joblib"
# DEFAULT_MODEL_NAME = "baseline_rf_model"
# DEFAULT_MODEL_VERSION = "2.0.0"
# DEFAULT_MODEL_DESCRIPTION = "Baseline Random Forest Model"
# DEFAULT_MODEL_AUTHOR = "Nitin S"
# DEFAULT_MODEL_EMAIL = "nitins1@gmail.com"
# DEFAULT_MODEL_DATE = datetime.now().strftime("%Y-%m-%d")
# DEFAULT_MODEL_TIME = datetime.now().strftime("%H:%M:%S")
# DEFAULT_MODEL_TIMEZONE = "UTC"
# DEFAULT_MODEL_TIMEZONE_OFFSET = 0
