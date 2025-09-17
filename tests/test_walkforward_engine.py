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

    series_ir = walkforward._information_ratio_frame(
        pd.Series({"beta": 0.5, "alpha": 0.75}), cols
    )
    assert series_ir.loc["information_ratio", "beta"] == 0.5


def test_infer_periods_per_year_edge_cases():
    idx_monthly = pd.date_range("2020-01-31", periods=4, freq="ME")
    assert walkforward._infer_periods_per_year(idx_monthly) == 12

    idx_weekly = pd.date_range("2020-01-03", periods=4, freq="7D")
    assert walkforward._infer_periods_per_year(idx_weekly) == 52

    idx_daily = pd.date_range("2020-01-01", periods=300, freq="B")
    assert walkforward._infer_periods_per_year(idx_daily) == 252

    idx_negative = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-01"])
    assert walkforward._infer_periods_per_year(idx_negative) == 1

    idx_sparse = pd.to_datetime(
        ["2020-01-01", "2022-12-31", "2025-12-31", "2028-12-31"]
    )
    assert walkforward._infer_periods_per_year(idx_sparse) == 1


def test_prepare_index_converts_and_validates():
    df = pd.DataFrame(
        {
            "Date": ["2020-02-01", "2020-01-01"],
            "alpha": [2, 1],
        }
    )
    prepared = walkforward._prepare_index(df)
    assert list(prepared.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]
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
