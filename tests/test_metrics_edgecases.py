import pandas as pd
import numpy as np
import pytest
from trend_analysis import metrics


def test_available_metrics_and_empty_like():
    assert "annual_return" in metrics.available_metrics()
    df = pd.DataFrame({"a": [0.1]})
    res = metrics._empty_like(df, "foo")
    assert isinstance(res, pd.Series)
    assert res.isna().all()


def test_check_shapes_mismatch():
    s = pd.Series([0.1, 0.2])
    df = pd.DataFrame({"a": [0.1, 0.2]})
    with pytest.raises(ValueError):
        metrics._check_shapes(df, s, "oops")


def test_max_drawdown_type_error():
    with pytest.raises(TypeError):
        metrics.max_drawdown([0.1, -0.1])  # type: ignore[arg-type]


def test_information_ratio_edge_cases():
    short = pd.Series([0.1])
    assert np.isnan(metrics.information_ratio(short, benchmark=0.0))
    df = pd.DataFrame({"a": [0.02, 0.03], "b": [0.01, 0.02]})
    bench = pd.DataFrame({"m": [0.01, 0.01]})
    ir = metrics.information_ratio(df, bench)
    assert isinstance(ir, pd.Series) and set(ir.index) == {"a", "b"}


def test_information_ratio_default_and_scalar():
    df = pd.DataFrame({"a": [0.02, 0.03], "b": [0.01, 0.02]})
    ir_none = metrics.information_ratio(df)
    assert isinstance(ir_none, pd.Series)
    ir_scalar = metrics.information_ratio(df, benchmark=0.0)
    assert isinstance(ir_scalar, pd.Series)
