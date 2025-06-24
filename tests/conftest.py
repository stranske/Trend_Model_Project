import pathlib
import pandas as pd
import pytest

@pytest.fixture()
def parquet_data(tmp_path):
    """Return a DataFrame read from a temporary Parquet file."""
    csv_path = pathlib.Path(__file__).resolve().parent / "data" / "sample.csv"
    df = pd.read_csv(csv_path)
    pq_path = tmp_path / "sample.parquet"
    df.to_parquet(pq_path, index=False)
    return pd.read_parquet(pq_path)
