import numpy as np
import pandas as pd
import pytest

from trend_portfolio_app.monte_carlo.engine import (
    BlockBootstrapModel,
    ReturnModel,
    ReturnModelConfig,
)


def make_panel():
    data = np.arange(60).reshape(20, 3) + 1
    dates = pd.date_range("2020-01-31", periods=20, freq="ME")
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C"])


def test_sample_before_fit_raises():
    model = BlockBootstrapModel(ReturnModelConfig(block=3, seed=0))
    with pytest.raises(RuntimeError):
        model.sample(5, 1)


def test_return_model_base_methods_raise():
    model = ReturnModel()
    with pytest.raises(NotImplementedError):
        model.fit(pd.DataFrame())
    with pytest.raises(NotImplementedError):
        model.sample(5, 1)


def test_output_shape_and_reproducibility():
    panel = make_panel()
    cfg = ReturnModelConfig(block=4, seed=42)
    model1 = BlockBootstrapModel(cfg)
    model1.fit(panel)
    out1 = model1.sample(8, 2)
    assert out1.shape == (2, 8, panel.shape[1])

    model2 = BlockBootstrapModel(cfg)
    model2.fit(panel)
    out2 = model2.sample(8, 2)
    np.testing.assert_array_equal(out1, out2)

    cfg_diff = ReturnModelConfig(block=4, seed=43)
    model3 = BlockBootstrapModel(cfg_diff)
    model3.fit(panel)
    out3 = model3.sample(8, 2)
    assert not np.array_equal(out1, out3)


def test_stitching_with_non_divisible_periods():
    panel = make_panel()
    cfg = ReturnModelConfig(block=6, seed=1)
    model = BlockBootstrapModel(cfg)
    model.fit(panel)
    out = model.sample(10, 1)
    assert out.shape == (1, 10, panel.shape[1])
    assert (out != 0).all()
