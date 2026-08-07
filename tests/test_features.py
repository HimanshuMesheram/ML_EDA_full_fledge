import numpy as np

from src.config import BINARY_FEATURES, CATEGORICAL_FEATURES, DROP_FEATURES, NUMERICAL_FEATURES
from src.features import build_preprocessing_pipeline, execute_feature_engineering


def test_execute_feature_engineering(sample_raw_dataframe):
    """Test feature engineering transforms and indicator flag creation."""
    df_fe = execute_feature_engineering(sample_raw_dataframe)

    # Check new engineered columns exist
    assert 'property_value_isna' in df_fe.columns
    assert 'dtir1_isna' in df_fe.columns
    assert 'income_isna' in df_fe.columns
    assert 'LTV_isna' in df_fe.columns
    assert 'Est_Monthly_Payment' in df_fe.columns
    assert 'Payment_to_Income' in df_fe.columns
    assert 'Loan_to_Income' in df_fe.columns
    assert 'LTV_calculated' in df_fe.columns
    assert 'DTI_x_LTV' in df_fe.columns
    assert 'age_ordinal' in df_fe.columns
    assert 'total_units_ordinal' in df_fe.columns

    # Check indicator flag logic for missing values
    assert df_fe.loc[2, 'property_value_isna'] == 1  # 3rd row has np.nan property_value
    assert df_fe.loc[0, 'property_value_isna'] == 0

def test_preprocessing_pipeline(sample_raw_dataframe):
    """Test scikit-learn preprocessing pipeline fitting and transformation."""
    df_fe = execute_feature_engineering(sample_raw_dataframe)

    # Drop identified non-useful columns
    cols_to_drop = [c for c in DROP_FEATURES if c in df_fe.columns]
    df_clean = df_fe.drop(columns=cols_to_drop)

    X = df_clean.drop(columns=['Status'])

    num_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
    bin_cols = [c for c in BINARY_FEATURES if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols, bin_cols)
    X_trans = preprocessor.fit_transform(X)

    assert X_trans.shape[0] == len(sample_raw_dataframe)
    assert not np.isnan(X_trans).any()  # All nulls imputed
