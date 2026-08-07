import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_target_distribution(df: pd.DataFrame, target_col: str = 'Status'):
    """Plot target class distribution countplot and pie chart."""
    counts = df[target_col].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.countplot(data=df, x=target_col, ax=axes[0], palette=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Target Class Counts', fontsize=12, fontweight='bold')
    axes[0].set_xticklabels(['Non-Default (0)', 'Default (1)'])

    axes[1].pie(counts, labels=['Non-Default (0)', 'Default (1)'], autopct='%1.2f%%',
                colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[1].set_title('Target Class Proportions', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig

def plot_missingness_correlation(missing_df: pd.DataFrame):
    """Plot missing percentage and default rate by missingness condition."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.barplot(data=missing_df, x='Missing_Pct', y='Feature', ax=axes[0], palette='viridis')
    axes[0].set_title('Missing Percentage by Feature', fontsize=12, fontweight='bold')

    missing_df_melted = pd.melt(missing_df.head(8), id_vars=['Feature'],
                                value_vars=['Default_Rate_If_Missing (%)', 'Default_Rate_If_Present (%)'],
                                var_name='Condition', value_name='Default_Rate')
    sns.barplot(data=missing_df_melted, x='Default_Rate', y='Feature', hue='Condition', ax=axes[1], palette='Set1')
    axes[1].set_title('Default Rate: Missing vs Present', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig

def plot_correlation_matrix(df: pd.DataFrame, num_cols: list):
    """Plot Pearson and Spearman correlation heatmaps."""
    df_sub = df[num_cols].dropna()
    corr_p = df_sub.corr(method='pearson')
    corr_s = df_sub.corr(method='spearman')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    mask = np.triu(np.ones_like(corr_p, dtype=bool))

    sns.heatmap(corr_p, annot=True, fmt='.2f', mask=mask, cmap='coolwarm', ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title('Pearson Linear Correlation', fontsize=12, fontweight='bold')

    sns.heatmap(corr_s, annot=True, fmt='.2f', mask=mask, cmap='coolwarm', ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title('Spearman Monotonic Correlation', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig
