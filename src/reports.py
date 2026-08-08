import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURES_DIR, REPORTS_DIR, TARGET_COL, TEST_PROCESSED_PATH
from src.logger import get_logger
from pathlib import Path

logger = get_logger(__name__)


def generate_evaluation_reports(model_path, model_name: str = "model"):
    """
    Generate model evaluation metrics and figures for a specific model.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_reports_dir = REPORTS_DIR / model_name
    figures_dir = model_reports_dir / "figures"
    model_reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not TEST_PROCESSED_PATH.exists() or not Path(model_path).exists():
        raise FileNotFoundError("Processed test data or model artifact missing. Run train first.")

    df_test = pd.read_csv(TEST_PROCESSED_PATH)
    model = joblib.load(model_path)

    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL]

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.50).astype(int)

    # Calculate metrics
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    metrics_summary = {
        'ROC_AUC': round(float(roc_auc), 4),
        'PR_AUC': round(float(pr_auc), 4),
        'Accuracy': round(float(acc), 4),
        'Precision': round(float(prec), 4),
        'Recall': round(float(rec), 4),
        'F1_Score': round(float(f1), 4),
        'Confusion_Matrix': {
            'True_Negative': int(cm[0, 0]),
            'False_Positive': int(cm[0, 1]),
            'False_Negative': int(cm[1, 0]),
            'True_Positive': int(cm[1, 1])
        }
    }

    # Save metrics JSON
    metrics_path = model_reports_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info("Saved evaluation metrics to: '%s'", metrics_path)

    # 1. Save ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#2980b9', lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#7f8c8d', linestyle='--')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    plt.tight_layout()
    roc_fig_path = figures_dir / "roc_curve.png"
    fig.savefig(roc_fig_path, dpi=300)
    plt.close(fig)

    # 2. Save PR Curve
    p_pts, r_pts, _ = precision_recall_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(r_pts, p_pts, color='#27ae60', lw=2, label=f'{model_name} (PR-AUC = {pr_auc:.4f})')
    ax.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower left')
    plt.tight_layout()
    pr_fig_path = figures_dir / "pr_curve.png"
    fig.savefig(pr_fig_path, dpi=300)
    plt.close(fig)

    # 3. Save Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Non-Default (0)', 'Default (1)'],
                yticklabels=['Non-Default (0)', 'Default (1)'])
    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    cm_fig_path = figures_dir / "confusion_matrix.png"
    fig.savefig(cm_fig_path, dpi=300)
    plt.close(fig)

    # 4. Save Feature Importance Plot
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_test.columns
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(data=fi_df.head(15), x='Importance', y='Feature', hue='Feature', palette='mako', legend=False, ax=ax)
        ax.set_title('Top 15 Feature Importances (Random Forest)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        fi_fig_path = figures_dir / "feature_importance.png"
        fig.savefig(fi_fig_path, dpi=300)
        plt.close(fig)

    logger.info("Generated evaluation figures in: '%s'", figures_dir)

if __name__ == '__main__':
    print("This module provides report generation utilities. Use from code.")
