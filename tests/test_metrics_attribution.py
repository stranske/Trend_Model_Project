import matplotlib.pyplot as plt
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


def test_compute_contributions_sums_to_total():
    signals, rebal = _sample_data()
    contrib = attribution.compute_contributions(signals, rebal)
    expected_total = signals.sum(axis=1) + rebal
    assert np.allclose(contrib["total"], expected_total)
    assert np.allclose(
        contrib.drop(columns="total").sum(axis=1), contrib["total"], atol=1e-9
    )


def test_compute_contributions_index_mismatch():
    signals, rebal = _sample_data()
    shifted = rebal.copy()
    shifted.index = shifted.index + 1

    with pytest.raises(ValueError, match="Indexes of signal_pnls"):
        attribution.compute_contributions(signals, shifted)


def test_compute_contributions_total_mismatch(monkeypatch):
    signals, rebal = _sample_data()

    def fake_allclose(*_, **__):
        return False

    monkeypatch.setattr(attribution.np, "allclose", fake_allclose)

    with pytest.raises(ValueError, match="Contributions do not sum"):
        attribution.compute_contributions(signals, rebal)


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


def test_plot_contributions_with_axis_sequence_and_labels():
    signals, rebal = _sample_data()
    contrib = attribution.compute_contributions(signals, rebal)

    # Use labels to focus on a subset and pass an ndarray of axes to hit the
    # sequence-handling branch.
    fig, axes = plt.subplots(2)
    try:
        returned_ax = attribution.plot_contributions(
            contrib, ax=axes, labels=["s1", "rebalancing"]
        )
        assert returned_ax is axes[0]
    finally:
        plt.close(fig)
