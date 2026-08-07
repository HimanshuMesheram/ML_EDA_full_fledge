"""
Loan Default Prediction Inference API Endpoint
"""

import json
from typing import Any, Dict

import joblib
import pandas as pd

from src.config import MODEL_PATH
from src.features import execute_feature_engineering


class LoanDefaultPredictor:
    """Production predictor wrapper for online or batch loan default scoring."""

    def __init__(self, model_path=MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}. Train model first.")
        self.model = joblib.load(model_path)

    def predict_single(self, application_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single application dictionary.
        
        Parameters
        ----------
        application_dict : dict
            Input dictionary of borrower attributes.
            
        Returns
        -------
        dict
            Prediction probabilities and approval decision recommendation.
        """
        df_single = pd.DataFrame([application_dict])
        df_fe = execute_feature_engineering(df_single)

        # Align features with expected model input
        if hasattr(self.model, 'feature_names_in_'):
            expected_cols = self.model.feature_names_in_
            for col in expected_cols:
                if col not in df_fe.columns:
                    df_fe[col] = 0
            df_input = df_fe[expected_cols]
        else:
            df_input = df_fe

        prob = float(self.model.predict_proba(df_input)[0, 1])
        decision = "REJECT / HIGH RISK" if prob >= 0.50 else "APPROVE / LOW RISK"

        return {
            "default_probability": round(prob, 4),
            "underwriting_recommendation": decision,
            "risk_score": int((1 - prob) * 850)
        }

if __name__ == '__main__':
    predictor = LoanDefaultPredictor()
    sample_application = {
        'loan_amount': 300000,
        'property_value': 400000,
        'income': 7500,
        'term': 360,
        'dtir1': 42.0,
        'LTV': 75.0,
        'loan_type': 'type1',
        'loan_purpose': 'p1',
        'occupancy_type': 'pr',
        'submission_of_application': 'to_inst',
        'Neg_ammortization': 'not_neg',
        'lump_sum_payment': 'not_lpsm',
        'co-applicant_credit_type': 'CIB',
        'age': '35-44',
        'total_units': '1U'
    }

    res = predictor.predict_single(sample_application)
    print("="*60)
    print("SAMPLE API INFERENCE RESPONSE")
    print("="*60)
    print(json.dumps(res, indent=2))
    print("="*60)
