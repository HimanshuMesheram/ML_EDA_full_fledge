import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    DROP_FEATURES,
    NUMERICAL_FEATURES,
    PROCESSED_DATA_DIR,
    TARGET_COL,
    TEST_PROCESSED_PATH,
    TRAIN_PROCESSED_PATH,
)
from src.dataset import load_raw_data


def execute_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute production domain feature engineering.
    
    Creates:
    1. Missingness indicator flags for pre-decision features.
    2. Calculated financial leverage and debt capacity ratios.
    3. Ordinal integer encodings.
    """
    df_fe = df.copy()

    # 1. Missingness Indicator Flags
    df_fe['property_value_isna'] = df_fe['property_value'].isnull().astype(int) if 'property_value' in df_fe else 0
    df_fe['dtir1_isna'] = df_fe['dtir1'].isnull().astype(int) if 'dtir1' in df_fe else 0
    df_fe['income_isna'] = df_fe['income'].isnull().astype(int) if 'income' in df_fe else 0
    df_fe['LTV_isna'] = df_fe['LTV'].isnull().astype(int) if 'LTV' in df_fe else 0

    # Helpers for safe division
    income_monthly = (df_fe['income'] / 12.0).replace(0, np.nan) if 'income' in df_fe else np.nan
    term_months = df_fe['term'].replace(0, np.nan) if 'term' in df_fe else np.nan

    # 2. Calculated Financial Ratios
    if 'loan_amount' in df_fe and 'term' in df_fe:
        df_fe['Est_Monthly_Payment'] = df_fe['loan_amount'] / term_months
        if 'income' in df_fe:
            df_fe['Payment_to_Income'] = df_fe['Est_Monthly_Payment'] / income_monthly
            df_fe['Loan_to_Income'] = df_fe['loan_amount'] / (df_fe['income'] + 1.0)

    if 'loan_amount' in df_fe and 'property_value' in df_fe:
        df_fe['LTV_calculated'] = (df_fe['loan_amount'] / (df_fe['property_value'] + 1.0)) * 100.0

    if 'dtir1' in df_fe and 'LTV' in df_fe:
        df_fe['DTI_x_LTV'] = df_fe['dtir1'] * df_fe['LTV']

    # 3. Ordinal Encodings
    if 'age' in df_fe:
        age_map = {'<25': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55-64': 5, '65-74': 6, '>74': 7}
        df_fe['age_ordinal'] = df_fe['age'].map(age_map).fillna(3)

    if 'total_units' in df_fe:
        units_map = {'1U': 1, '2U': 2, '3U': 3, '4U': 4}
        df_fe['total_units_ordinal'] = df_fe['total_units'].map(units_map).fillna(1)

    return df_fe

def build_preprocessing_pipeline(num_cols: list, cat_cols: list, bin_cols: list) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer preprocessing pipeline.
    """
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    bin_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols),
        ('bin', bin_transformer, bin_cols)
    ])

    return preprocessor

def create_processed_datasets():
    """
    Load raw data, run feature engineering, fit pipeline on X_train, and export train/test artifacts.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw dataset...")
    df_raw = load_raw_data()

    print("Executing feature engineering...")
    df_engineered = execute_feature_engineering(df_raw)

    # Drop identified non-useful and leakage features
    cols_to_drop = [c for c in DROP_FEATURES if c in df_engineered.columns]
    df_clean = df_engineered.drop(columns=cols_to_drop)

    # Separate features and target
    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]

    # Filter available column lists
    num_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
    bin_cols = [c for c in BINARY_FEATURES if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols, bin_cols)

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Fitting preprocessing pipeline on training split...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Extract feature names
    cat_onehot_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
    final_feature_names = num_cols + list(cat_onehot_names) + bin_cols

    # Build processed DataFrames
    df_train_proc = pd.DataFrame(X_train_proc, columns=final_feature_names)
    df_train_proc[TARGET_COL] = y_train.values

    df_test_proc = pd.DataFrame(X_test_proc, columns=final_feature_names)
    df_test_proc[TARGET_COL] = y_test.values

    # Save CSV artifacts
    df_train_proc.to_csv(TRAIN_PROCESSED_PATH, index=False)
    df_test_proc.to_csv(TEST_PROCESSED_PATH, index=False)

    print("="*80)
    print("FEATURE ENGINEERING & PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Train Artifact: '{TRAIN_PROCESSED_PATH}' ({df_train_proc.shape[0]:,} rows x {df_train_proc.shape[1]} cols)")
    print(f"Test Artifact:  '{TEST_PROCESSED_PATH}' ({df_test_proc.shape[0]:,} rows x {df_test_proc.shape[1]} cols)")
    print("="*80)

if __name__ == '__main__':
    create_processed_datasets()
