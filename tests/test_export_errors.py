import pandas as pd
import pytest

from trend_analysis.export import export_data


def test_export_data_bad_format(tmp_path):
    data = {"s": pd.DataFrame({"A": [1]})}
    with pytest.raises(ValueError):
        export_data(data, str(tmp_path / "out"), formats=["bad"])

