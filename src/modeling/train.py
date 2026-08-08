import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src import reports
from src.config import MODELS_DIR, TARGET_COL, TEST_PROCESSED_PATH, TRAIN_PROCESSED_PATH
from src.logger import get_logger
from src.models_manager import save_model

logger = get_logger(__name__)


def train_baseline_models():
    """
    Load processed train/test datasets, train baseline classifiers, evaluate performance, and serialize model.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed dataset artifacts...")
    if not TRAIN_PROCESSED_PATH.exists() or not TEST_PROCESSED_PATH.exists():
        from src.features import create_processed_datasets
        create_processed_datasets()

    df_train = pd.read_csv(TRAIN_PROCESSED_PATH)
    df_test = pd.read_csv(TEST_PROCESSED_PATH)

    X_tr = df_train.drop(columns=[TARGET_COL])
    y_tr = df_train[TARGET_COL]

    X_te = df_test.drop(columns=[TARGET_COL])
    y_te = df_test[TARGET_COL]

    # 1. Train Logistic Regression
    logger.info("Training Logistic Regression Classifier...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_model.fit(X_tr, y_tr)
    lr_preds = lr_model.predict_proba(X_te)[:, 1]

    lr_auc = roc_auc_score(y_te, lr_preds)
    lr_prauc = average_precision_score(y_te, lr_preds)

    # 2. Train Random Forest
    logger.info("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42, class_weight='balanced', n_jobs=-1
    )
    rf_model.fit(X_tr, y_tr)
    rf_preds = rf_model.predict_proba(X_te)[:, 1]

    # 3. Try training XGBoost if available
    xgb_model = None
    try:
        import xgboost as xgb
        logger.info("Training XGBoost Classifier...")
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb_model.fit(X_tr, y_tr)
        xgb_preds = xgb_model.predict_proba(X_te)[:, 1]
        xgb_auc = roc_auc_score(y_te, xgb_preds)
        xgb_prauc = average_precision_score(y_te, xgb_preds)
    except Exception as e:
        logger.warning(f"XGBoost not available or failed to train: {e}")
        xgb_model = None

    rf_auc = roc_auc_score(y_te, rf_preds)
    rf_prauc = average_precision_score(y_te, rf_preds)

    logger.info("%s", "="*80)
    logger.info("MODEL TRAINING & EVALUATION RESULTS")
    logger.info("%s", "="*80)
    logger.info("Logistic Regression | Test ROC-AUC: %.4f | Test PR-AUC: %.4f", lr_auc, lr_prauc)
    logger.info("Random Forest       | Test ROC-AUC: %.4f | Test PR-AUC: %.4f", rf_auc, rf_prauc)
    if xgb_model is not None:
        logger.info("XGBoost             | Test ROC-AUC: %.4f | Test PR-AUC: %.4f", xgb_auc, xgb_prauc)
    logger.info("%s", "="*80)

    # Serialize models with versioning and save metadata
    lr_path = save_model(lr_model, "logistic_regression", metadata={"roc_auc": lr_auc, "pr_auc": lr_prauc})
    logger.info("Saved Logistic Regression model to: %s", lr_path)

    rf_path = save_model(rf_model, "random_forest", metadata={"roc_auc": rf_auc, "pr_auc": rf_prauc})
    logger.info("Saved Random Forest model to: %s", rf_path)

    if xgb_model is not None:
        xgb_path = save_model(xgb_model, "xgboost", metadata={"roc_auc": xgb_auc, "pr_auc": xgb_prauc})
        logger.info("Saved XGBoost model to: %s", xgb_path)

    # Generate per-model reports
    try:
        reports.generate_evaluation_reports(model_path=lr_path, model_name='logistic_regression')
        reports.generate_evaluation_reports(model_path=rf_path, model_name='random_forest')
        if xgb_model is not None:
            reports.generate_evaluation_reports(model_path=xgb_path, model_name='xgboost')
    except Exception as e:
        logger.exception("Failed to generate reports: %s", e)

if __name__ == '__main__':
    train_baseline_models()
