import numpy as np
import pandas as pd
import pytest

from trend_analysis.metrics import attribution


def _sample_data():
    signals = pd.DataFrame(
        {
            "s1": [0.01, 0.02, -0.01],
            "s2": [0.00, -0.01, 0.02],
        },
        index=pd.RangeIndex(3),
    )
    rebal = pd.Series([0.001, -0.002, 0.0], index=signals.index, name="rebalancing")
    return signals, rebal


def _zero_exposure_data():
    signals = pd.DataFrame(0.0, index=pd.RangeIndex(3), columns=["s1", "s2"])
    rebal = pd.Series(0.0, index=signals.index, name="rebalancing")
    return signals, rebal


def test_compute_contributions_sums_to_total():
    signals, rebal = _sample_data()
    contrib = attribution.compute_contributions(signals, rebal)
    expected_total = signals.sum(axis=1) + rebal
    assert np.allclose(contrib["total"], expected_total)
    assert np.allclose(
        contrib.drop(columns="total").sum(axis=1), contrib["total"], atol=1e-9
    )


def test_export_and_plot(tmp_path):
    signals, rebal = _sample_data()
    contrib = attribution.compute_contributions(signals, rebal)

    # export
    out = tmp_path / "contrib.csv"
    attribution.export_contributions(contrib, out)
    loaded = pd.read_csv(out, index_col=0)
    pd.testing.assert_frame_equal(loaded, contrib)

    # plot
    ax = attribution.plot_contributions(contrib)
    assert hasattr(ax, "plot")


def test_zero_exposure_handled():
    signals, rebal = _zero_exposure_data()
    contrib = attribution.compute_contributions(signals, rebal)
    assert (contrib == 0).all().all()


def test_index_mismatch_raises():
    signals, rebal = _sample_data()
    rebal = rebal.reindex(rebal.index[::-1])
    with pytest.raises(ValueError):
        attribution.compute_contributions(signals, rebal)
