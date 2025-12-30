import numpy as np
import pandas as pd

from trend_analysis.plugins import create_weight_engine


def test_risk_parity_simple():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=["a", "b"], columns=["a", "b"])
    engine = create_weight_engine("risk_parity")
    w = engine.weight(cov)
    expected = pd.Series([0.6, 0.4], index=["a", "b"])
    assert np.allclose(w.values, expected.values)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()


def test_equal_risk_contribution_balances_rc():
    cov = pd.DataFrame([[0.04, 0.024], [0.024, 0.09]], index=["a", "b"], columns=["a", "b"])
    engine = create_weight_engine("erc")
    w = engine.weight(cov)
    mrc = cov.values @ w.values
    rc = w.values * mrc
    assert np.isclose(rc[0], rc[1], rtol=1e-6)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()


def test_hierarchical_risk_parity_expected_weights():
    cov = pd.DataFrame(
        [[0.04, 0.006, 0.0], [0.006, 0.09, 0.0], [0.0, 0.0, 0.16]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    engine = create_weight_engine("hrp")
    w = engine.weight(cov)
    expected = pd.Series([0.582234, 0.258771, 0.158995], index=["a", "b", "c"])
    assert np.allclose(w.values, expected.values, atol=1e-6)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
