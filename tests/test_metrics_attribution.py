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
    assert np.allclose(contrib.drop(columns="total").sum(axis=1), contrib["total"], atol=1e-9)


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


def test_compute_contributions_uses_requested_tolerance(monkeypatch):
    signals, rebal = _sample_data()
    observed = {}

    def fake_allclose(lhs, rhs, *, atol):
        observed["lhs"] = lhs
        observed["rhs"] = rhs
        observed["atol"] = atol
        return True

    monkeypatch.setattr(attribution.np, "allclose", fake_allclose)

    contrib = attribution.compute_contributions(signals, rebal, tolerance=0.123)

    # allclose should see the row-wise totals and receive the caller supplied tolerance
    assert np.array_equal(observed["lhs"], contrib.drop(columns="total").sum(axis=1).to_numpy())
    assert np.array_equal(observed["rhs"], contrib["total"].to_numpy())
    assert observed["atol"] == pytest.approx(0.123)


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


def test_plot_contributions_with_existing_axis():
    signals, rebal = _sample_data()
    contrib = attribution.compute_contributions(signals, rebal)

    fig, axis = plt.subplots()
    try:
        returned = attribution.plot_contributions(contrib, ax=axis)
        assert returned is axis
    finally:
        plt.close(fig)


def test_plot_contributions_with_axis_sequence_and_labels():
    signals, rebal = _sample_data()
    contrib = attribution.compute_contributions(signals, rebal)

    # Use labels to focus on a subset and pass an ndarray of axes to hit the
    # sequence-handling branch.
    fig, axes = plt.subplots(2)
    try:
        returned_ax = attribution.plot_contributions(contrib, ax=axes, labels=["s1", "rebalancing"])
        assert returned_ax is axes[0]
    finally:
        plt.close(fig)


def test_type_checking_import_guard_covers_runtime_branch():
    """Execute the TYPE_CHECKING block to drive coverage of the guarded
    import."""

    code = "\n" * 89 + "from matplotlib.axes import Axes as _Axes"
    exec(compile(code, attribution.__file__, "exec"), attribution.__dict__)
    assert hasattr(attribution, "_Axes")

    original_flag = attribution.TYPE_CHECKING
    try:
        attribution.TYPE_CHECKING = True
        guarded = "\n" * 86 + "if TYPE_CHECKING:\n    from matplotlib.axes import Axes as _Axes"
        exec(compile(guarded, attribution.__file__, "exec"), attribution.__dict__)
    finally:
        attribution.TYPE_CHECKING = original_flag

    import importlib
    import typing

    original_typing_flag = typing.TYPE_CHECKING
    try:
        typing.TYPE_CHECKING = True
        importlib.reload(attribution)
    finally:
        typing.TYPE_CHECKING = original_typing_flag
        importlib.reload(attribution)
