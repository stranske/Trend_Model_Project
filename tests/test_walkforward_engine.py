import builtins

import numpy as np
import pandas as pd
import pytest

from trend_analysis.engine import walkforward


def _monthly_frame(periods: int = 8) -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=periods, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "alpha": np.linspace(0.01, 0.08, periods),
            "beta": np.linspace(0.015, 0.12, periods),
        }
    )


def test_walk_forward_builds_regime_tables():
    df = _monthly_frame()
    regimes = pd.Series(
        ["bull", "bear", "bull", "bear", "bull", "bear", "bull", "bear"],
        index=df["Date"],
    )

    result = walkforward.walk_forward(
        df,
        train_size=3,
        test_size=2,
        step_size=2,
        agg={"alpha": ["mean", "std"], "beta": ["mean", "max"]},
        regimes=regimes,
    )

    # Monthly data should infer 12 periods per year via the 10-14 branch.
    assert result.periods_per_year == 12
    # Ensure the regime table retains the MultiIndex structure and includes values.
    assert not result.by_regime.empty
    assert result.by_regime.columns.names == ["metric", "statistic"]
    # Window metadata should appear before metric columns and maintain ordering.
    assert result.oos_windows.columns[0] == ("window", "train_start")
    assert ("alpha", "information_ratio") in result.oos_windows.columns


def test_walk_forward_records_scalar_information_ratio(monkeypatch):
    df = _monthly_frame(5)

    def fake_ir(frame, benchmark=0.0, periods_per_year=12):  # noqa: ARG001
        return 1.5

    monkeypatch.setattr(walkforward, "information_ratio", fake_ir)

    result = walkforward.walk_forward(
        df,
        train_size=2,
        test_size=2,
        step_size=1,
        metric_cols=["alpha"],
    )

    # The scalar information ratio should be mapped to the only metric column.
    assert ("alpha", "information_ratio") in result.oos_windows.columns
    assert result.oos_windows[("alpha", "information_ratio")].notna().any()


def test_walkforward_helper_functions_cover_branches():
    df = pd.DataFrame({"x": [1, 2]}, index=pd.Index(["row1", "row2"], name="idx"))
    as_df = walkforward._to_dataframe(df)
    assert list(as_df.index) == ["row1", "row2"]

    series = pd.Series([1, 2], name="vals")
    as_df_from_series = walkforward._to_dataframe(series)
    assert list(as_df_from_series.index) == ["vals"]

    with pytest.raises(TypeError):
        walkforward._to_dataframe([1, 2])

    empty_flat = walkforward._flatten_agg_result(pd.DataFrame())
    assert empty_flat.empty

    cols = pd.Index(["alpha", "beta"])
    scalar_ir = walkforward._information_ratio_frame(0.25, cols)
    assert scalar_ir.iloc[0, 0] == 0.25
    assert np.isnan(scalar_ir.iloc[0, 1])

    series_ir = walkforward._information_ratio_frame(pd.Series({"beta": 0.5, "alpha": 0.75}), cols)
    assert series_ir.loc["information_ratio", "beta"] == 0.5


def test_infer_periods_per_year_edge_cases():
    idx_monthly = pd.date_range("2020-01-31", periods=4, freq="ME")
    assert walkforward._infer_periods_per_year(idx_monthly) == 12

    idx_weekly = pd.date_range("2020-01-03", periods=4, freq="7D")
    assert walkforward._infer_periods_per_year(idx_weekly) == 52

    idx_daily = pd.date_range("2020-01-01", periods=300, freq="B")
    assert walkforward._infer_periods_per_year(idx_daily) == 252

    idx_quarterly = pd.date_range("2020-03-31", periods=4, freq="QE")
    assert walkforward._infer_periods_per_year(idx_quarterly) == 4

    idx_negative = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-01"])
    assert walkforward._infer_periods_per_year(idx_negative) == 1

    idx_sparse = pd.to_datetime(["2020-01-01", "2022-12-31", "2025-12-31", "2028-12-31"])
    assert walkforward._infer_periods_per_year(idx_sparse) == 1


def test_infer_periods_per_year_returns_general_case():
    """Values outside the preset windows should return the rounded estimate."""

    # Roughly bi-weekly cadence → approx ≈ 26, none of the special cases apply.
    idx = pd.date_range("2020-01-01", periods=10, freq="14D")
    assert walkforward._infer_periods_per_year(idx) == pytest.approx(26)


def test_prepare_index_converts_and_validates():
    df = pd.DataFrame(
        {
            "Date": ["2020-02-01", "2020-01-01"],
            "alpha": [2, 1],
        }
    )
    prepared = walkforward._prepare_index(df)
    assert list(prepared.index) == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-02-01"),
    ]
    assert prepared.index.is_monotonic_increasing

    with pytest.raises(ValueError):
        walkforward._prepare_index(pd.DataFrame({"alpha": [1, 2]}))


def test_walk_forward_regime_rows_include_information_ratio(monkeypatch):
    df = _monthly_frame(6)

    def fake_ir(frame, benchmark=0.0, periods_per_year=12):  # noqa: ARG001
        return pd.Series({col: idx + 1.0 for idx, col in enumerate(frame.columns)})

    monkeypatch.setattr(walkforward, "information_ratio", fake_ir)

    regimes = pd.Series(
        ["bull", "bear", None, "bull", "bear", "bull"],
        index=pd.to_datetime(df["Date"]),
    )

    result = walkforward.walk_forward(
        df,
        train_size=3,
        test_size=2,
        step_size=1,
        agg={"alpha": ["mean", "max"], "beta": ["mean"]},
        regimes=regimes,
    )

    # Regime aggregation should yield non-empty rows with MultiIndex columns
    assert not result.by_regime.empty
    assert set(result.by_regime.columns.get_level_values("statistic")) >= {
        "mean",
        "information_ratio",
    }
    # Out-of-sample windows should be enumerated and ordered
    assert list(result.oos_windows.index) == sorted(result.oos_windows.index)
    assert result.oos_windows.columns[0][0] == "window"


def test_information_ratio_frame_handles_empty_columns():
    cols = pd.Index([])
    frame = walkforward._information_ratio_frame(0.5, cols)
    assert frame.empty
    assert list(frame.index) == ["information_ratio"]


def test_agg_label_prefers_callable_name():
    def custom_metric(values):  # noqa: ARG001
        return values.mean()

    class _CallableWithoutName:
        def __call__(self, values):  # noqa: D401, ANN001
            return float(values.mean())

    assert walkforward._agg_label("mean") == "mean"
    assert walkforward._agg_label(custom_metric) == "custom_metric"
    assert walkforward._agg_label(object()) == "value"
    # Callable instances without a ``__name__`` attribute should fall back to
    # the generic label.
    assert walkforward._agg_label(_CallableWithoutName()) == "value"


def test_infer_periods_per_year_branch_guards(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=5, freq="D")

    assert walkforward._infer_periods_per_year(idx[:1]) == 1

    real_diff = np.diff
    monkeypatch.setattr(np, "diff", lambda arr: np.array([], dtype=np.int64))
    assert walkforward._infer_periods_per_year(idx) == 1
    monkeypatch.setattr(np, "diff", real_diff, raising=False)

    real_median = np.median
    monkeypatch.setattr(np, "median", lambda arr: -10)
    assert walkforward._infer_periods_per_year(idx) == 1

    class _PositiveMedian:
        def __le__(self, other):  # noqa: D401, ANN001
            return False

        def __truediv__(self, other):  # noqa: D401, ANN001
            return 0

    monkeypatch.setattr(np, "median", lambda arr: _PositiveMedian())
    assert walkforward._infer_periods_per_year(idx) == 1
    monkeypatch.setattr(np, "median", real_median, raising=False)


def test_walk_forward_handles_empty_metric_columns():
    df = _monthly_frame(5)

    result = walkforward.walk_forward(
        df,
        train_size=2,
        test_size=1,
        step_size=1,
        metric_cols=[],
    )

    # Without metric columns the OOS aggregates should be empty but metadata retained.
    assert result.oos.empty
    assert list(result.oos_windows.columns.get_level_values("category")) == [
        "window",
        "window",
        "window",
        "window",
        "window",
        "window",
    ]


def test_walk_forward_scalar_ir_without_metrics(monkeypatch):
    df = _monthly_frame(6)
    regimes = pd.Series(
        ["bull", "bear", "bull", "bear", "bull", "bear"],
        index=pd.to_datetime(df["Date"]),
    )

    flagged: set[int] = set()

    def fake_ir(frame, benchmark=0.0, periods_per_year=12):  # noqa: ARG001
        flagged.add(id(frame.columns))
        return 0.42

    monkeypatch.setattr(walkforward, "information_ratio", fake_ir)
    monkeypatch.setattr(
        walkforward,
        "_information_ratio_frame",
        lambda *a, **k: pd.DataFrame(),
    )
    monkeypatch.setattr(
        walkforward,
        "_to_dataframe",
        lambda *a, **k: pd.DataFrame(),
    )
    monkeypatch.setattr(
        walkforward,
        "_flatten_agg_result",
        lambda *a, **k: pd.Series(dtype=float),
    )
    monkeypatch.setattr(
        walkforward,
        "len",
        lambda obj, _orig=builtins.len: (  # type: ignore[arg-type]
            0 if (obj_id := id(obj)) in flagged and not flagged.remove(obj_id) else _orig(obj)
        ),
        raising=False,
    )

    result = walkforward.walk_forward(
        df,
        train_size=3,
        test_size=1,
        step_size=1,
        metric_cols=["alpha", "beta"],
        regimes=regimes,
    )

    # No metric columns survive, so scalar information ratios are ignored and
    # regime aggregation yields an empty table rather than raising.
    # The columns of result.oos_windows are expected to be a pandas MultiIndex
    # with at least two levels: (window, metric_name). This assertion checks that
    # no metric column named "information_ratio" survives, except for the "window" columns.
    assert all(
        col[1] != "information_ratio" for col in result.oos_windows.columns if col[0] != "window"
    )
    assert result.by_regime.empty


def test_walk_forward_no_splits_yields_empty_windows():
    df = _monthly_frame(3)

    result = walkforward.walk_forward(
        df,
        train_size=5,
        test_size=3,
        step_size=1,
    )

    assert result.splits == []
    assert result.oos_windows.empty


def test_walk_forward_ignores_missing_regime_labels():
    df = _monthly_frame(6)
    # All regimes are missing/NaN which should skip regime aggregation block
    regimes = pd.Series([None] * len(df), index=pd.to_datetime(df["Date"]))

    result = walkforward.walk_forward(
        df,
        train_size=3,
        test_size=2,
        step_size=1,
        regimes=regimes,
    )

    assert result.by_regime.empty


def test_walk_forward_orders_oos_window_columns():
    df = _monthly_frame(8).set_index("Date")
    # Reverse column order to ensure sorting branch executes.
    df = df[["beta", "alpha"]]

    result = walkforward.walk_forward(
        df,
        train_size=4,
        test_size=2,
        step_size=2,
        agg={"beta": ["mean", "max"], "alpha": ["std"]},
    )

    assert not result.oos_windows.empty
    window_cols = [col for col in result.oos_windows.columns if col[0] == "window"]
    metric_cols = [col for col in result.oos_windows.columns if col[0] != "window"]
    # Metric columns should be sorted alphabetically by metric/statistic tuple.
    assert metric_cols == sorted(metric_cols, key=lambda c: (str(c[0]), str(c[1])))
    # The window metadata columns stay at the front of the frame.
    assert window_cols[0] == ("window", "train_start")


def test_walk_forward_handles_empty_test_windows(monkeypatch):
    df = _monthly_frame(4)
    regimes = pd.Series(["r1", "r2", "r3", "r4"], index=df["Date"])

    base_index = pd.to_datetime(df["Date"])
    custom_split = walkforward.Split(
        train_start=base_index[0],
        train_end=base_index[1],
        test_start=pd.NaT,
        test_end=pd.NaT,
        train_index=pd.DatetimeIndex(base_index[:2]),
        test_index=pd.DatetimeIndex([]),
    )

    monkeypatch.setattr(
        walkforward,
        "_generate_splits",
        lambda *args, **kwargs: [custom_split],
    )

    result = walkforward.walk_forward(
        df,
        train_size=2,
        test_size=1,
        step_size=1,
        regimes=regimes,
    )

    # With empty test windows no OOS regimes should be produced.
    assert result.by_regime.empty
    # Information ratio columns are omitted when no test data is present.
    assert all("information_ratio" not in col for col in result.oos_windows.columns)
    # The OOS aggregation dataframe should contain only NaN placeholders.
    assert result.oos.isna().all().all()
