import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import MODEL_PATH, MODELS_DIR, TARGET_COL, TEST_PROCESSED_PATH, TRAIN_PROCESSED_PATH


def train_baseline_models():
    """
    Load processed train/test datasets, train baseline classifiers, evaluate performance, and serialize model.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed dataset artifacts...")
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
    print("Training Logistic Regression Classifier...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_model.fit(X_tr, y_tr)
    lr_preds = lr_model.predict_proba(X_te)[:, 1]

    lr_auc = roc_auc_score(y_te, lr_preds)
    lr_prauc = average_precision_score(y_te, lr_preds)

    # 2. Train Random Forest
    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42, class_weight='balanced', n_jobs=-1
    )
    rf_model.fit(X_tr, y_tr)
    rf_preds = rf_model.predict_proba(X_te)[:, 1]

    rf_auc = roc_auc_score(y_te, rf_preds)
    rf_prauc = average_precision_score(y_te, rf_preds)

    print("="*80)
    print("MODEL TRAINING & EVALUATION RESULTS")
    print("="*80)
    print(f"Logistic Regression | Test ROC-AUC: {lr_auc:.4f} | Test PR-AUC: {lr_prauc:.4f}")
    print(f"Random Forest       | Test ROC-AUC: {rf_auc:.4f} | Test PR-AUC: {rf_prauc:.4f}")
    print("="*80)

    # Serialize Random Forest model
    joblib.dump(rf_model, MODEL_PATH)
    print(f"Serialized model artifact saved to: '{MODEL_PATH}'")

if __name__ == '__main__':
    train_baseline_models()
