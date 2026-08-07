import pathlib

import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COL


def load_raw_data(data_path: pathlib.Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw Loan Default CSV dataset.
    
    Parameters
    ----------
    data_path : pathlib.Path
        Path to the raw CSV file.
        
    Returns
    -------
    pd.DataFrame
        Loaded raw dataset.
    """
    if not data_path.exists():
        # Fallback check in current directory
        fallback = pathlib.Path("Loan_Default.csv")
        if fallback.exists():
            data_path = fallback
        else:
            raise FileNotFoundError(f"Raw dataset file not found at {data_path} or {fallback}")

    df = pd.read_csv(data_path)
    return df

def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a structural overview table for the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
        
    Returns
    -------
    pd.DataFrame
        Summary table detailing column types, null counts, unique cardinalities, and sample values.
    """
    summary_list = []
    for col in df.columns:
        summary_list.append({
            'Column_Name': col,
            'Data_Type': str(df[col].dtype),
            'Non_Null_Count': df[col].notnull().sum(),
            'Missing_Count': df[col].isnull().sum(),
            'Missing_Pct': round(df[col].isnull().mean() * 100, 2),
            'Unique_Values': df[col].nunique(),
            'Sample_Value': str(df[col].dropna().iloc[0]) if df[col].notnull().any() else 'None'
        })
    return pd.DataFrame(summary_list)

if __name__ == '__main__':
    print("="*60)
    print("DATASET VERIFICATION & LOADING")
    print("="*60)
    df_raw = load_raw_data()
    print(f"Dataset Shape: {df_raw.shape[0]:,} rows | {df_raw.shape[1]} columns")
    print(f"Target Column ('{TARGET_COL}') Value Counts:")
    print(df_raw[TARGET_COL].value_counts(normalize=True).round(4) * 100)
    print("="*60)
