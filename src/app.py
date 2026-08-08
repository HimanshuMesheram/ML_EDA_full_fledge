import joblib
import pandas as pd
from flask import Flask, jsonify, request

from src.features import execute_feature_engineering
from src.logger import get_logger
from src.models_manager import get_latest_model_path, get_model_by_version

logger = get_logger(__name__)
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint. JSON body should be application dict. Optional query params: model_name, model_version."""
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    model_name = request.args.get('model_name', 'random_forest')
    model_version = request.args.get('model_version', None)

    # Resolve model path
    model_path = None
    if model_version:
        model_path = get_model_by_version(model_name, model_version)
    else:
        model_path = get_latest_model_path(model_name)

    if model_path is None:
        logger.error("Model not found: %s (version=%s)", model_name, model_version)
        return jsonify({"error": "Model not found"}), 404

    model = joblib.load(model_path)

    # Feature engineering
    df_single = pd.DataFrame([data])
    df_fe = execute_feature_engineering(df_single)

    if hasattr(model, 'feature_names_in_'):
        expected_cols = model.feature_names_in_
        for col in expected_cols:
            if col not in df_fe.columns:
                df_fe[col] = 0
        df_input = df_fe[expected_cols]
    else:
        df_input = df_fe

    prob = float(model.predict_proba(df_input)[0, 1])
    decision = "REJECT / HIGH RISK" if prob >= 0.50 else "APPROVE / LOW RISK"

    resp = {
        "default_probability": round(prob, 4),
        "underwriting_recommendation": decision,
        "risk_score": int((1 - prob) * 850),
        "model_used": str(model_path)
    }
    return jsonify(resp)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
