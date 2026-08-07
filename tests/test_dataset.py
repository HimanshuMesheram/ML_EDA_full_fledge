import pandas as pd

from src.config import TARGET_COL
from src.dataset import dataset_summary, load_raw_data


def test_load_raw_data_exists():
    """Verify that raw dataset loads successfully and has expected shape."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df.shape[1] == 34
    assert TARGET_COL in df.columns

def test_dataset_summary(sample_raw_dataframe):
    """Verify dataset summary profiling table output."""
    summary = dataset_summary(sample_raw_dataframe)
    assert isinstance(summary, pd.DataFrame)
    assert 'Column_Name' in summary.columns
    assert 'Missing_Count' in summary.columns
    assert 'Missing_Pct' in summary.columns
    assert len(summary) == sample_raw_dataframe.shape[1]
