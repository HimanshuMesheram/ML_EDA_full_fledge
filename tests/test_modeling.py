
import pandas as pd
import pytest

from src.config import MODEL_PATH, TEST_PROCESSED_PATH
from src.modeling.predict import predict_default_probability


def test_prediction_output():
    """Verify that batch prediction outputs valid probability bounds [0, 1]."""
    if not TEST_PROCESSED_PATH.exists() or not MODEL_PATH.exists():
        pytest.skip("Processed test data or model artifact missing")

    results = predict_default_probability(data_path=TEST_PROCESSED_PATH, model_path=MODEL_PATH)

    assert isinstance(results, pd.DataFrame)
    assert 'Predicted_Default_Prob' in results.columns
    assert 'Predicted_Status' in results.columns

    # Probabilities must be strictly bounded between 0 and 1
    probs = results['Predicted_Default_Prob']
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Predicted status must be binary {0, 1}
    preds = results['Predicted_Status']
    assert set(preds.unique()).issubset({0, 1})
