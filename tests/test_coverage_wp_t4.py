"""Targeted coverage tests for multi_period engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest
from trend.config_schema import CoreConfigError
from trend_analysis.config import Config
from trend_analysis.multi_period import engine as mp_engine
from trend_analysis.multi_period import run as run_mp
from trend_analysis.multi_period.engine import (
    MembershipWindow,
    _as_weight_series,
    _build_rebalance_frame,
    _canonical_benchmark_columns,
    _configured_custom_weighting,
    _enforce_max_active_positions,
    _mapping_or_attr_get,
    _membership_table_from_frame,
    _parse_positive_int,
    _PeriodSetup,
    _resolve_max_turnover_cap,
    _resolve_portfolio_weighting,
    _resolve_regime_label_for_window,
    _resolve_regime_proxy_column,
    _setup_period,
    _zscore_or_zero,
)
from trend_analysis.regimes import normalise_settings
from trend_analysis.weighting import EqualWeight


def test_canonical_benchmark_columns_without_available_columns() -> None:
    benchmarks = {"b1": "SPY", "b2": "AGG"}
    res = _canonical_benchmark_columns(benchmarks, available_columns=None)
    assert res == ("SPY", "AGG")


def test_resolve_portfolio_weighting_invalid_params() -> None:
    cfg = {"weighting": {"name": "equal", "params": "not_a_mapping"}}
    with pytest.raises(ValueError, match="^portfolio.weighting.params must be a mapping$"):
        _resolve_portfolio_weighting(cfg)


def test_configured_custom_weighting_validation() -> None:
    holdings = ["FundA", "FundB"]

    with pytest.raises(
        ValueError,
        match="^portfolio.weighting.name 'custom' requires portfolio.custom_weights$",
    ):
        _configured_custom_weighting({}, holdings)

    with pytest.raises(ValueError, match="^portfolio.custom_weights.FundA must be numeric$"):
        _configured_custom_weighting({"custom_weights": {"FundA": "invalid"}}, holdings)

    with pytest.raises(ValueError, match="^portfolio.custom_weights.FundA must be finite$"):
        _configured_custom_weighting({"custom_weights": {"FundA": float("inf")}}, holdings)

    valid_cfg = {"custom_weights": {"FundA": 0.3, "FundB": 0.7}}
    df = _configured_custom_weighting(valid_cfg, holdings)
    assert df.to_dict()["weight"] == {"FundA": pytest.approx(0.3), "FundB": pytest.approx(0.7)}


def test_mapping_or_attr_get_mapping_and_object() -> None:
    mapping_source = {"alpha": 10, "beta": 20}
    assert _mapping_or_attr_get(mapping_source, "alpha", 0) == 10
    assert _mapping_or_attr_get(mapping_source, "gamma", 99) == 99

    class DummyObj:
        def __init__(self) -> None:
            self.delta = 42

    obj = DummyObj()
    assert _mapping_or_attr_get(obj, "delta", 0) == 42
    assert _mapping_or_attr_get(obj, "epsilon", 100) == 100


def test_setup_period_cooldown_decrement() -> None:
    @dataclass
    class DummyPt:
        out_end: str = "2020-03-31"
        in_start: str = "2019-01-01"
        in_end: str = "2019-12-31"
        out_start: str = "2020-01-01"

    cooldown_book = {"FundA": 1, "FundB": 3}

    def dummy_valid(
        df: pd.DataFrame,
        in_start: str,
        in_end: str,
        out_start: str,
        out_end: str,
        *,
        require_out_sample: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
        return pd.DataFrame(), pd.DataFrame(), ["FundB"], ""

    setup = _setup_period(
        DummyPt(),
        cooldown_periods=2,
        cooldown_book=cooldown_book,
        valid_universe=dummy_valid,
        df=pd.DataFrame(),
    )
    assert isinstance(setup, _PeriodSetup)
    assert setup.period_ts == pd.Timestamp("2020-03-31")
    assert "FundA" not in cooldown_book
    assert cooldown_book["FundB"] == 2


def test_resolve_regime_proxy_column_lookup() -> None:
    cols = ["SPY_ETF", "AGG_BOND", "cash"]
    bench = {"benchmark_alias": "SPY_ETF", "AGG_ALIAS": "AGG_BOND"}

    assert _resolve_regime_proxy_column("SPY_ETF", cols, bench) == "SPY_ETF"
    assert _resolve_regime_proxy_column("benchmark_alias", cols, bench) == "SPY_ETF"
    assert _resolve_regime_proxy_column("agg_bond", cols, bench) == "AGG_BOND"
    assert _resolve_regime_proxy_column("CASH", cols, None) == "cash"
    assert _resolve_regime_proxy_column("NONEXISTENT", cols, bench) is None


def test_resolve_regime_label_for_window_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    in_df = pd.DataFrame(
        {"SPY": [0.01, 0.02, -0.01], "ALL_NAN": [np.nan, np.nan, np.nan]}, index=dates
    )

    settings_disabled = normalise_settings({"enabled": False, "proxy": "SPY"})
    assert (
        _resolve_regime_label_for_window(
            in_df,
            regime_settings=settings_disabled,
            benchmarks_cfg=None,
            regime_frequency="M",
            regime_ppy=12.0,
        )
        is None
    )

    settings_missing_proxy = normalise_settings({"enabled": True, "proxy": "NONEXISTENT"})
    assert (
        _resolve_regime_label_for_window(
            in_df,
            regime_settings=settings_missing_proxy,
            benchmarks_cfg=None,
            regime_frequency="M",
            regime_ppy=12.0,
        )
        is None
    )

    settings_nan = normalise_settings({"enabled": True, "proxy": "ALL_NAN"})
    assert (
        _resolve_regime_label_for_window(
            in_df,
            regime_settings=settings_nan,
            benchmarks_cfg=None,
            regime_frequency="M",
            regime_ppy=12.0,
        )
        is None
    )

    settings_empty_labels = normalise_settings({"enabled": True, "proxy": "SPY"})
    monkeypatch.setattr(mp_engine, "compute_regimes", lambda *a, **k: pd.Series([], dtype=object))
    assert (
        _resolve_regime_label_for_window(
            in_df,
            regime_settings=settings_empty_labels,
            benchmarks_cfg=None,
            regime_frequency="M",
            regime_ppy=12.0,
        )
        is None
    )


def test_resolve_max_turnover_cap_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    in_df = pd.DataFrame({"SPY": [0.01, -0.01, 0.02]}, index=dates)
    reg_settings = normalise_settings({"enabled": True, "proxy": "SPY"})

    monkeypatch.setattr(mp_engine, "parse_regime_turnover_caps", lambda *a, **k: None)
    with pytest.raises(CoreConfigError, match="max_turnover must be numeric scalars"):
        _resolve_max_turnover_cap(
            in_df,
            max_turnover_cfg=1.0,
            regime_settings=reg_settings,
            benchmarks_cfg=None,
            regime_frequency="M",
            regime_ppy=12.0,
        )

    monkeypatch.setattr(mp_engine, "_resolve_regime_label_for_window", lambda *a, **k: "normal")
    monkeypatch.setattr(mp_engine, "parse_regime_turnover_caps", lambda *a, **k: {"normal": 0.5})
    monkeypatch.setattr(mp_engine, "_resolve_turnover_cap_from_parsed", lambda *a, **k: None)
    cap = _resolve_max_turnover_cap(
        in_df,
        max_turnover_cfg={"normal": 0.5},
        regime_settings=reg_settings,
        benchmarks_cfg=None,
        regime_frequency="M",
        regime_ppy=12.0,
    )
    assert cap == 1.0


def test_membership_table_from_frame_parsing() -> None:
    assert _membership_table_from_frame(None) == {}
    assert _membership_table_from_frame(pd.DataFrame()) == {}

    with pytest.raises(
        ValueError,
        match="^membership data must include fund, effective_date, and end_date columns$",
    ):
        _membership_table_from_frame(pd.DataFrame({"invalid": [1, 2]}))

    raw = pd.DataFrame(
        {
            "fund": ["FundA", "FundA", "FundB"],
            "effective_date": ["2020-01-01", "2020-07-01", "2020-02-01"],
            "end_date": ["2020-06-30", None, "2020-12-31"],
        }
    )
    table = _membership_table_from_frame(raw)
    assert sorted(table.keys()) == ["FundA", "FundB"]
    assert table["FundA"] == (
        MembershipWindow(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30")),
        MembershipWindow(pd.Timestamp("2020-07-01"), None),
    )
    assert table["FundB"] == (
        MembershipWindow(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-12-31")),
    )

    unsorted = pd.DataFrame(
        {
            "fund": [" FundC ", " FundC ", "FundD"],
            "effective_date": [
                "2020-10-1",
                "2020-2-1",
                pd.Timestamp("2020-03-01"),
            ],
            "end_date": ["2020-12-31", "2020-06-30", "2020-08-31"],
        }
    )
    fund_c = _membership_table_from_frame(unsorted)["FundC"]
    assert fund_c[0].effective_date == pd.Timestamp("2020-02-01")
    assert fund_c[0].end_date == pd.Timestamp("2020-06-30")
    assert fund_c[1].effective_date == pd.Timestamp("2020-10-01")
    assert fund_c[1].end_date == pd.Timestamp("2020-12-31")
    assert _membership_table_from_frame(unsorted)["FundD"] == (
        MembershipWindow(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-08-31")),
    )


def test_as_weight_series_inputs() -> None:
    df_no_weight_col = pd.DataFrame({"score": [0.4, 0.6]}, index=["A", "B"])
    res1 = _as_weight_series(df_no_weight_col)
    assert res1.to_dict() == {"A": 0.4, "B": 0.6}

    s = pd.Series({"A": 0.2, "B": 0.8})
    res2 = _as_weight_series(s)
    assert res2.to_dict() == {"A": 0.2, "B": 0.8}

    lst = [0.5, 0.5]
    res3 = _as_weight_series(lst)
    assert list(res3) == [0.5, 0.5]


def test_enforce_max_active_positions_protected() -> None:
    weights = pd.Series({"FundA": 0.1, "FundB": 0.5, "FundC": 0.4})
    unchanged = _enforce_max_active_positions(weights, max_active_positions=0)
    assert unchanged.to_dict() == weights.to_dict()

    trimmed = _enforce_max_active_positions(
        weights,
        max_active_positions=2,
        protected=["FundA"],
    )
    assert trimmed["FundA"] == pytest.approx(1.0 / 6.0)
    assert trimmed["FundB"] == pytest.approx(5.0 / 6.0)
    assert trimmed["FundC"] == 0.0
    assert trimmed.sum() == pytest.approx(1.0)


def test_zscore_or_zero_flat_series() -> None:
    flat = pd.Series([2.5, 2.5, 2.5], index=["X", "Y", "Z"])
    z = _zscore_or_zero(flat)
    assert z.to_dict() == {"X": 0.0, "Y": 0.0, "Z": 0.0}


def test_build_rebalance_frame_schedule_and_fallback() -> None:
    class DummyCfg:
        def __init__(self) -> None:
            self.portfolio = {"rebalance_freq": "M"}
            self.multi_period = {"in_sample_len": 1}

    cfg = DummyCfg()

    out_dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    out_df = pd.DataFrame({"FundA": [0.01, 0.02]}, index=out_dates)

    full_dates = pd.date_range("2019-01-31", periods=15, freq="ME")
    df_indexed = pd.DataFrame(
        {"FundA": [0.01] * 15, "FundB": [0.02] * 15},
        index=full_dates,
    )

    effective_w = pd.Series({"FundA": 0.6, "FundB": 0.4})

    def fake_score_frame(window: pd.DataFrame, holdings: list[str], **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({"Sharpe": [1.0, 0.5]}, index=holdings)

    def fake_ensure_zscore(sf: pd.DataFrame, metric: str) -> pd.DataFrame:
        sf_out = sf.copy()
        sf_out["zscore"] = [1.0, -1.0]
        return sf_out

    def fake_apply_policy(weights_df: pd.DataFrame, signals: pd.Series | None) -> pd.Series:
        return weights_df["weight"]

    # Test normal rolling frame execution
    rf = _build_rebalance_frame(
        cfg,
        out_df=out_df,
        df_indexed=df_indexed,
        realised_holdings=["FundA", "FundB"],
        effective_w=effective_w,
        rf_override=None,
        periods_per_year=12,
        metric="Sharpe",
        weighting=EqualWeight(),
        weighting_scheme="equal",
        use_risk_weighting=False,
        risk_weight_engine=None,
        min_w_bound=0.05,
        max_w_bound=0.8,
        max_active_positions=None,
        min_tenure_guard=set(),
        score_frame_fn=fake_score_frame,
        ensure_zscore_fn=fake_ensure_zscore,
        apply_policy_to_weights_fn=fake_apply_policy,
    )
    assert rf is not None
    assert len(rf) == 2
    assert rf.iloc[0].to_dict() == {"FundA": pytest.approx(0.5), "FundB": pytest.approx(0.5)}

    # Test empty window fallback where window.empty leads to w_row = prev_reb_w
    df_empty_window = pd.DataFrame(columns=["FundA", "FundB"], index=pd.DatetimeIndex([]))
    rf_empty = _build_rebalance_frame(
        cfg,
        out_df=out_df,
        df_indexed=df_empty_window,
        realised_holdings=["FundA", "FundB"],
        effective_w=effective_w,
        rf_override=None,
        periods_per_year=12,
        metric="Sharpe",
        weighting=EqualWeight(),
        weighting_scheme="equal",
        use_risk_weighting=False,
        risk_weight_engine=None,
        min_w_bound=0.05,
        max_w_bound=0.8,
        max_active_positions=None,
        min_tenure_guard=set(),
        score_frame_fn=fake_score_frame,
        ensure_zscore_fn=fake_ensure_zscore,
        apply_policy_to_weights_fn=fake_apply_policy,
    )
    assert rf_empty is not None
    assert rf_empty.iloc[0].to_dict() == {"FundA": pytest.approx(0.6), "FundB": pytest.approx(0.4)}


def test_parse_positive_int_branches() -> None:
    assert _parse_positive_int("abc") is None
    assert _parse_positive_int(object()) is None
    assert _parse_positive_int(None) is None
    assert _parse_positive_int(0) is None
    assert _parse_positive_int(-3) is None
    assert _parse_positive_int(7) == 7
    assert _parse_positive_int("12") == 12


def test_run_empty_value_columns_raises() -> None:
    class DummyCfg:
        def __init__(self) -> None:
            self.data = {"missing_policy": "ffill"}
            self.portfolio = {"cost_model": {}}
            self.multi_period = {"frequency": "M"}

    df = pd.DataFrame({"Date": pd.date_range("2020-01-31", periods=3, freq="ME")})
    with pytest.raises(ValueError, match="^Missing-data policy removed all assets for analysis$"):
        mp_engine.run(DummyCfg(), df)


def test_run_resamples_daily_data_to_monthly(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, pd.DataFrame] = {}

    def fake_run_mp(cfg: Any, df: pd.DataFrame, **kwargs: Any) -> list[str]:
        captured["df"] = df
        return ["ok"]

    monkeypatch.setattr(mp_engine, "_run_threshold_hold_multi_periods", fake_run_mp)

    class DummyCfg:
        def __init__(self) -> None:
            self.data = {"missing_policy": "ffill"}
            self.portfolio = {"policy": "threshold_hold", "cost_model": {}}
            self.multi_period = {"frequency": "M"}

    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    df = pd.DataFrame({"Date": dates, "FundA": [0.01] * 40})
    res = mp_engine.run(DummyCfg(), df)
    assert res == ["ok"]
    assert "df" in captured
    cdf = captured["df"]
    # 23 business days in January 2020 compounded: 1.01^23 - 1
    expected_jan = (1.01**23) - 1.0
    assert cdf["FundA"].iloc[0] == pytest.approx(expected_jan, rel=1e-6)


def test_run_mapping_missing_policy_preserves_periods(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, pd.DataFrame] = {}

    def fake_run_mp(cfg: Any, df: pd.DataFrame, **kwargs: Any) -> list[str]:
        captured["df"] = df
        return ["ok"]

    monkeypatch.setattr(mp_engine, "_run_threshold_hold_multi_periods", fake_run_mp)

    class DummyCfg:
        def __init__(self, policy: Any) -> None:
            self.data = {"missing_policy": policy}
            self.portfolio = {"policy": "threshold_hold", "cost_model": {}}
            self.multi_period = {"frequency": "M"}

    # An all-NaN month with all-ffill mapping keeps row 2 when preserve_empty_periods is True (pins line 1914)
    df_empty_month = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "FundA": [0.05, np.nan],
            "FundB": [0.01, np.nan],
        }
    )
    mp_engine.run(DummyCfg({"FundA": "ffill", "FundB": "ffill"}), df_empty_month)
    assert len(captured["df"]) == 2
    assert captured["df"]["Date"].tolist() == [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-29"),
    ]

    captured.clear()
    mixed_cfg = DummyCfg({"FundA": "ffill", "FundB": "drop"})
    mp_engine.run(mixed_cfg, df_empty_month)
    assert len(captured["df"]) == 2
    assert captured["df"]["Date"].tolist() == [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-29"),
    ]


def test_run_inception_date_masking(monkeypatch: pytest.MonkeyPatch) -> None:
    import trend_analysis.data

    def fake_inc(df: pd.DataFrame) -> dict[str, pd.Timestamp | None]:
        return {
            "FundA": pd.Timestamp("2020-02-29"),
            "FundB": None,
            "FundC_absent": pd.Timestamp("2020-01-31"),
        }

    monkeypatch.setattr(trend_analysis.data, "compute_inception_dates", fake_inc)

    captured: dict[str, pd.DataFrame] = {}

    def fake_run_mp(cfg: Any, df: pd.DataFrame, **kwargs: Any) -> list[str]:
        captured["df"] = df
        return ["ok"]

    monkeypatch.setattr(mp_engine, "_run_threshold_hold_multi_periods", fake_run_mp)

    class DummyCfg:
        def __init__(self) -> None:
            self.data = {"missing_policy": "ffill"}
            self.portfolio = {"policy": "threshold_hold", "cost_model": {}}
            self.multi_period = {"frequency": "M"}

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "FundA": [0.01, 0.02, 0.03],
            "FundB": [0.04, 0.05, 0.06],
        }
    )
    mp_engine.run(DummyCfg(), df)
    cdf = captured["df"]
    assert "FundB" not in cdf.columns
    assert list(cdf.columns) == ["Date", "FundA"]
    assert pd.isna(cdf.loc[cdf["Date"] < "2020-02-29", "FundA"].iloc[0])
    assert cdf.attrs.get("inception_dates") == {
        "FundA": "2020-02-29",
        "FundB": None,
        "FundC_absent": "2020-01-31",
    }


def test_run_membership_required_but_missing_raises() -> None:
    class DummyCfg:
        def __init__(self) -> None:
            self.data = {"missing_policy": "ffill", "universe_membership_path": "mem.csv"}
            self.portfolio = {"policy": "threshold_hold", "cost_model": {}}
            self.multi_period = {"frequency": "M"}

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "FundA": [0.01, 0.02],
        }
    )
    with pytest.raises(
        ValueError,
        match="^cfg.data\\['universe_membership_path'\\] was provided; pass a membership",
    ):
        mp_engine.run(DummyCfg(), df, membership=None)


def test_run_membership_missing_entries_raises() -> None:
    class DummyCfg:
        def __init__(self) -> None:
            self.data = {"missing_policy": "ffill"}
            self.portfolio = {"policy": "threshold_hold", "cost_model": {}}
            self.multi_period = {"frequency": "M"}

    funds = {f"Fund{i:02d}": [0.01, 0.02] for i in range(1, 8)}
    funds["Date"] = pd.to_datetime(["2020-01-31", "2020-02-29"])
    df = pd.DataFrame(funds)
    mem = pd.DataFrame(
        {
            "fund": ["KnownFund"],
            "effective_date": ["2020-01-01"],
            "end_date": ["2020-12-31"],
        }
    )
    with pytest.raises(
        ValueError,
        match=(
            r"^Universe membership is missing effective_date entries for: "
            r"Fund01, Fund02, Fund03, Fund04, Fund05…$"
        ),
    ):
        mp_engine.run(DummyCfg(), df, membership=mem)


def test_run_membership_removes_all_rows_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyCfg:
        def __init__(self) -> None:
            self.data = {"missing_policy": "ffill"}
            self.portfolio = {"policy": "threshold_hold", "cost_model": {}}
            self.multi_period = {"frequency": "M"}

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "FundA": [0.01, 0.02],
            "FundB": [0.03, 0.04],
        }
    )
    # Effective window in 2025, completely disjoint from 2020
    mem = pd.DataFrame(
        {
            "fund": ["FundA", "FundB"],
            "effective_date": ["2025-01-01", "2025-01-01"],
            "end_date": ["2025-12-31", "2025-12-31"],
        }
    )
    with pytest.raises(ValueError, match="^Universe membership removed all rows for analysis$"):
        mp_engine.run(DummyCfg(), df, membership=mem)

    good_mem = pd.DataFrame(
        {
            "fund": ["FundA", "FundB"],
            "effective_date": ["2020-01-01", "2020-01-01"],
            "end_date": ["2020-12-31", "2020-12-31"],
        }
    )
    captured: dict[str, pd.DataFrame] = {}

    def fake_run_mp(cfg: Any, frame: pd.DataFrame, **kwargs: Any) -> list[str]:
        captured["df"] = frame
        return ["ok"]

    monkeypatch.setattr(mp_engine, "_run_threshold_hold_multi_periods", fake_run_mp)
    mp_engine.run(DummyCfg(), df, membership=good_mem)
    assert len(captured["df"]) == 2
    assert captured["df"]["FundA"].tolist() == [0.01, 0.02]
    assert captured["df"]["FundB"].tolist() == [0.03, 0.04]


def test_run_empty_membership_frame_treated_as_none() -> None:
    class DummyCfg:
        def __init__(self) -> None:
            self.data = {"missing_policy": "ffill", "universe_membership_path": "mem.csv"}
            self.portfolio = {"policy": "threshold_hold", "cost_model": {}}
            self.multi_period = {"frequency": "M"}

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "FundA": [0.01, 0.02],
        }
    )
    # Passing empty membership frame when membership is required raises because
    # line 1971 converts empty frame to None, causing line 1976 to raise
    with pytest.raises(
        ValueError,
        match="^cfg.data\\['universe_membership_path'\\] was provided; pass a membership",
    ):
        mp_engine.run(DummyCfg(), df, membership=pd.DataFrame())


def _make_base_config() -> Config:
    from pathlib import Path

    import yaml

    config_file = None
    for parent in Path(__file__).resolve().parents:
        cand = parent / "config" / "defaults.yml"
        if cand.is_file():
            config_file = cand
            break
    if config_file is None:
        raise FileNotFoundError("config/defaults.yml not found")

    cfg_data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    cfg_data["data"]["path"] = ""
    cfg_data["data"]["date_column"] = "Date"
    cfg_data["data"]["frequency"] = "M"
    cfg_data["data"]["allow_risk_free_fallback"] = True
    cfg_data["multi_period"] = {
        "frequency": "M",
        "in_sample_len": 2,
        "out_sample_len": 1,
        "start": "2020-01",
        "end": "2020-04",
    }
    cfg_data["portfolio"]["policy"] = "threshold_hold"
    cfg_data["portfolio"]["selection_mode"] = "rank"
    cfg_data["portfolio"]["rank"] = {"inclusion_approach": "top_n", "n": 2}
    cfg_data["portfolio"]["weighting"] = {"name": "equal"}
    cfg_data["portfolio"]["constraints"] = {
        "min_weight": 0.05,
        "max_weight": 0.7,
        "long_only": True,
    }
    return Config(**cfg_data)


def test_threshold_hold_invalid_config_parsing() -> None:
    cfg = _make_base_config()
    cfg.multi_period["min_history_periods"] = "bad"
    cfg.portfolio["threshold_hold"] = {"z_exit_hard": "bad", "metric": "Sharpe"}
    cfg.portfolio["rank"]["bottom_k"] = "bad"
    cfg.portfolio["constraints"]["min_funds"] = "bad"
    cfg.portfolio["constraints"]["max_active_positions"] = "bad"
    cfg.portfolio["constraints"]["min_weight_strikes"] = "bad"
    cfg.portfolio["cooldown_periods"] = "bad"
    cfg.portfolio["sticky_add_x"] = "bad"
    cfg.portfolio["sticky_drop_y"] = "bad"
    cfg.portfolio["min_tenure_n"] = -5

    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Alpha Fund": [0.05, 0.05, -0.02, 0.01],
            "Beta Fund": [0.01, -0.02, 0.06, -0.01],
            "Gamma Fund": [0.02, 0.03, -0.01, 0.04],
        }
    )
    cfg.portfolio["rank"]["n"] = 1
    res = run_mp(cfg, df)
    assert len(res) == 2
    assert list(res[0]["fund_weights"].keys()) == ["Alpha Fund"]
    cfg.portfolio["constraints"]["min_funds"] = 2
    res_min = run_mp(cfg, df)
    assert len(res_min[0]["fund_weights"]) == 2

    cfg_min_exit = _make_base_config()
    cfg_min_exit.portfolio["constraints"]["min_funds"] = "bad"
    cfg_min_exit.portfolio["threshold_hold"] = {
        "soft_strikes": 1,
        "z_exit_soft": 0.5,
        "z_entry_soft": 2.0,
    }
    cfg_min_exit.portfolio["rank"]["n"] = 1
    df_exit = pd.DataFrame(
        {
            "Date": dates,
            "Alpha Fund": [0.05, 0.05, -0.02, -0.02],
            "Beta Fund": [0.01, -0.02, 0.00, 0.00],
            "Gamma Fund": [0.02, 0.03, 0.00, 0.00],
        }
    )
    res_min_exit = run_mp(cfg_min_exit, df_exit)
    assert res_min_exit[1]["fund_weights"]["Gamma Fund"] == pytest.approx(0.62)

    hard_dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    hard_df = pd.DataFrame(
        {
            "Date": hard_dates,
            "RF": [0.001, 0.001, 0.001, 0.001],
            "FundA": [0.05, 0.06, -0.05, -0.05],
            "FundB": [0.04, 0.05, 0.04, 0.05],
            "FundC": [0.01, 0.02, 0.05, 0.06],
        }
    )
    cfg_bad_exit = _make_base_config()
    cfg_bad_exit.data["allow_risk_free_fallback"] = False
    cfg_bad_exit.data["risk_free_column"] = "RF"
    cfg_bad_exit.portfolio["threshold_hold"] = {"z_exit_hard": "bad", "metric": "Sharpe"}
    cfg_forced_exit = _make_base_config()
    cfg_forced_exit.data["allow_risk_free_fallback"] = False
    cfg_forced_exit.data["risk_free_column"] = "RF"
    cfg_forced_exit.portfolio["threshold_hold"] = {"z_exit_hard": -0.1, "metric": "Sharpe"}
    res_bad_exit = run_mp(cfg_bad_exit, hard_df)
    res_forced_exit = run_mp(cfg_forced_exit, hard_df)
    assert "FundA" in res_bad_exit[1]["fund_weights"]
    assert "FundA" not in res_forced_exit[1]["fund_weights"]


def test_threshold_hold_no_investable_funds() -> None:
    cfg = _make_base_config()
    cfg.data["allow_risk_free_fallback"] = False
    cfg.data["risk_free_column"] = "RF"
    cfg.portfolio["indices_list"] = ["Alpha Fund", "Beta Fund"]

    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RF": [0.001, 0.001, 0.001, 0.001],
            "Alpha Fund": [0.05, 0.05, -0.02, 0.01],
            "Beta Fund": [0.01, -0.02, 0.06, -0.01],
        }
    )
    res = run_mp(cfg, df)
    assert len(res) == 2
    assert res[0].get("fund_weights") == {}
    assert res[0].get("selected_funds") == []
    assert res[0]["turnover"] == 0.0
    assert res[0]["manager_changes"] == []
    assert res[0]["score_frame"].empty
    assert res[0]["in_sample_scaled"].empty
    assert res[0]["out_sample_scaled"].empty
    assert res[1].get("fund_weights") == {}
    assert res[1].get("selected_funds") == []
    assert res[1]["turnover"] == 0.0


def test_threshold_hold_missing_rf_series_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_base_config()
    cfg.metrics["rf_override_enabled"] = False
    monkeypatch.setattr(
        mp_engine, "_resolve_risk_free_column", lambda *a, **k: (None, ["Alpha Fund"], None)
    )

    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Alpha Fund": [0.01, 0.02, 0.03, 0.01],
        }
    )
    with pytest.raises(
        ValueError,
        match="^Risk-free override is disabled but no risk-free series is available",
    ):
        run_mp(cfg, df)


def test_threshold_hold_blended_metric_with_weights() -> None:
    cfg = _make_base_config()
    cfg.portfolio["threshold_hold"] = {
        "metric": "blended",
        "blended_weights": {"Sharpe": 0.5, "Sortino": 0.5},
    }
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Alpha Fund": [0.05, 0.05, -0.02, 0.01],
            "Beta Fund": [0.01, -0.02, 0.06, -0.01],
            "Gamma Fund": [0.02, 0.03, -0.01, 0.04],
        }
    )
    res = run_mp(cfg, df)
    assert len(res) == 2
    sf = res[0]["score_frame"]
    assert "blended" in sf.columns
    assert sf.loc["Alpha Fund", "blended"] == pytest.approx(0.5)


def test_threshold_hold_blended_metric_fallback_sharpe() -> None:
    cfg = _make_base_config()
    cfg.portfolio["threshold_hold"] = {
        "metric": "blended",
        "blended_weights": {},
    }
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Alpha Fund": [0.05, 0.05, -0.02, 0.01],
            "Beta Fund": [0.01, -0.02, 0.06, -0.01],
        }
    )
    res = run_mp(cfg, df)
    assert len(res) == 2
    ssf = res[0]["selection_score_frame"]
    assert "blended" in ssf.columns
    assert ssf.loc["Alpha Fund", "blended"] == pytest.approx(float(ssf.loc["Alpha Fund", "Sharpe"]))
    assert ssf.loc["Beta Fund", "blended"] == pytest.approx(float(ssf.loc["Beta Fund", "Sharpe"]))


def test_threshold_hold_random_mode() -> None:
    cfg = _make_base_config()
    cfg.seed = 42
    cfg.multi_period["end"] = "2020-05"
    cfg.portfolio["selection_mode"] = "random"
    cfg.portfolio["constraints"]["max_funds"] = 2
    cfg.portfolio["constraints"]["min_funds"] = 2
    cfg.portfolio["constraints"]["max_weight"] = 0.8

    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Alpha One": [0.05, 0.05, -0.02, 0.01, 0.02],
            "Alpha Two": [0.01, -0.02, 0.06, -0.01, 0.01],
            "Beta One": [0.02, 0.03, -0.01, 0.04, -0.02],
            "Beta Two": [-0.01, 0.04, 0.02, 0.02, 0.03],
            "Gamma One": [0.03, -0.01, 0.01, 0.03, 0.01],
        }
    )
    res = run_mp(cfg, df)
    assert len(res) == 3
    assert set(res[0]["fund_weights"].keys()) == {"Beta Two", "Gamma One"}
    assert res[0]["fund_weights"]["Beta Two"] == pytest.approx(0.5)
    assert res[0]["fund_weights"]["Gamma One"] == pytest.approx(0.5)
    assert set(res[1]["fund_weights"].keys()) == {"Alpha One", "Gamma One"}
    assert res[1]["fund_weights"]["Alpha One"] == pytest.approx(0.475)
    assert res[1]["fund_weights"]["Gamma One"] == pytest.approx(0.475)
    assert set(res[2]["fund_weights"].keys()) == {"Alpha Two", "Beta One"}


def test_threshold_hold_random_mode_seed7_single_fund_cap() -> None:
    cfg = _make_base_config()
    cfg.seed = 7
    cfg.multi_period["end"] = "2020-05"
    cfg.portfolio["selection_mode"] = "random"
    cfg.portfolio["constraints"]["max_funds"] = 1
    cfg.portfolio["rank"] = {"inclusion_approach": "top_n", "n": 3}
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "A1": [0.05, 0.05, -0.02, 0.01, 0.02],
            "A2": [0.01, -0.02, 0.06, -0.01, 0.01],
            "B1": [0.02, 0.03, -0.01, 0.04, -0.02],
            "B2": [-0.01, 0.04, 0.02, 0.02, 0.03],
            "C1": [0.03, -0.01, 0.01, 0.03, 0.01],
        }
    )
    res = run_mp(cfg, df)
    assert len(res) == 3
    assert [list(p["fund_weights"].keys()) for p in res] == [["A1"], ["A1"], ["A1"]]
    assert res[1]["fund_weights"]["A1"] == pytest.approx(0.7)


def test_threshold_hold_buy_and_hold_mode_and_data_ceased() -> None:
    cfg = _make_base_config()
    cfg.data["allow_risk_free_fallback"] = False
    cfg.data["risk_free_column"] = "RF"
    cfg.vol_adjust["enabled"] = False
    cfg.portfolio["policy"] = "threshold_hold"
    cfg.portfolio["selection_mode"] = "buy_and_hold"
    cfg.portfolio["rank"] = {"inclusion_approach": "top_n", "n": 2}
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RF": [0.001, 0.001, 0.001, 0.001],
            "Alpha Fund": [0.05, 0.06, 0.02, 0.01],
            "Beta Fund": [0.04, 0.05, 0.01, 0.01],
            "Gamma Fund": [0.01, 0.02, 0.03, 0.04],
        }
    )
    mem = pd.DataFrame(
        [
            {"fund": "Alpha Fund", "effective_date": "2020-01-01", "end_date": "2020-04-30"},
            {"fund": "Beta Fund", "effective_date": "2020-01-01", "end_date": "2020-03-31"},
            {"fund": "Gamma Fund", "effective_date": "2020-01-01", "end_date": "2020-04-30"},
            {"fund": "RF", "effective_date": "2020-01-01", "end_date": "2020-04-30"},
        ]
    )
    res = run_mp(cfg, df, membership=mem)
    assert len(res) == 2
    assert "Alpha Fund" in res[0]["fund_weights"]
    assert "Beta Fund" not in res[1]["fund_weights"]
    assert "Gamma Fund" in res[1]["fund_weights"]
    reasons = [e.get("reason") for e in res[1]["manager_changes"]]
    assert "replacement" in reasons
    assert "data_ceased" in reasons


def test_threshold_hold_sticky_add_and_drop() -> None:
    cfg = _make_base_config()
    cfg.data["allow_risk_free_fallback"] = False
    cfg.data["risk_free_column"] = "RF"
    cfg.vol_adjust["enabled"] = False
    cfg.portfolio["rank"] = {"inclusion_approach": "top_n", "n": 2}
    cfg.portfolio["threshold_hold"] = {"soft_strikes": 1, "z_exit_soft": 0.0}
    cfg.portfolio["constraints"]["min_funds"] = 2
    cfg.portfolio["sticky_add_x"] = 2

    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RF": [0.001, 0.001, 0.001, 0.001],
            "Alpha": [0.10, 0.12, -0.05, -0.05],
            "Beta": [0.08, 0.11, 0.08, 0.11],
            "Gamma": [0.01, 0.02, 0.09, 0.12],
        }
    )
    res = run_mp(cfg, df)
    assert len(res) == 2
    assert "Alpha" in res[0]["fund_weights"]
    assert "Beta" in res[0]["fund_weights"]
    assert "Beta" in res[1]["fund_weights"]
    assert "Gamma" not in res[1]["fund_weights"]


def test_threshold_hold_min_tenure_guard() -> None:
    cfg = _make_base_config()
    cfg.data["allow_risk_free_fallback"] = False
    cfg.data["risk_free_column"] = "RF"
    cfg.vol_adjust["enabled"] = False
    cfg.portfolio["rank"] = {"inclusion_approach": "top_n", "n": 2}
    cfg.portfolio["threshold_hold"] = {"soft_strikes": 1, "z_exit_soft": 0.0}
    cfg.portfolio["min_tenure_n"] = 2

    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RF": [0.001, 0.001, 0.001, 0.001],
            "Alpha": [0.10, 0.12, -0.05, -0.05],
            "Beta": [0.08, 0.11, 0.08, 0.11],
            "Gamma": [0.01, 0.02, 0.09, 0.12],
        }
    )
    res = run_mp(cfg, df)
    assert len(res) == 2
    assert "Alpha" in res[1]["fund_weights"]
    assert "Beta" in res[1]["fund_weights"]
    reasons = [e.get("reason") for e in res[1]["manager_changes"]]
    assert "min_tenure" in reasons


def test_threshold_hold_turnover_budget_max_changes() -> None:
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Alpha": [0.05, 0.05, -0.05, 0.05, 0.05],
            "Beta": [0.04, 0.04, 0.04, -0.05, -0.05],
            "Gamma": [0.01, 0.08, 0.08, 0.08, 0.08],
            "Delta": [0.01, 0.09, 0.09, 0.09, 0.09],
        }
    )

    cfg_unlimited = _make_base_config()
    cfg_unlimited.multi_period["end"] = "2020-05"
    cfg_unlimited.portfolio["constraints"]["max_funds"] = 4
    cfg_unlimited.portfolio["threshold_hold"] = {"z_entry_soft": 0.0}
    cfg_unlimited.portfolio["turnover_budget_max_changes"] = "bad"
    res_unlimited = run_mp(cfg_unlimited, df)
    assert len(res_unlimited) == 3
    assert len(res_unlimited[1]["fund_weights"]) == 4
    assert "Gamma" in res_unlimited[1]["fund_weights"]
    assert "Delta" in res_unlimited[1]["fund_weights"]
    budget_events = [
        ev
        for period in res_unlimited
        for ev in (period.get("manager_changes") or [])
        if ev.get("reason") == "turnover_budget"
    ]
    assert budget_events == []

    cfg_zero = _make_base_config()
    cfg_zero.multi_period["end"] = "2020-05"
    cfg_zero.portfolio["constraints"]["max_funds"] = 4
    cfg_zero.portfolio["threshold_hold"] = {"z_entry_soft": 0.0}
    cfg_zero.portfolio["turnover_budget_max_changes"] = 0
    res_zero = run_mp(cfg_zero, df)
    for period_bad, period_zero in zip(res_unlimited, res_zero, strict=True):
        assert period_bad["fund_weights"] == period_zero["fund_weights"]


def test_threshold_hold_hard_exit_forced() -> None:
    cfg = _make_base_config()
    cfg.data["risk_free_column"] = "RF"
    cfg.data["allow_risk_free_fallback"] = False
    cfg.portfolio["rank"] = {"inclusion_approach": "top_n", "n": 2}
    cfg.portfolio["threshold_hold"] = {"z_exit_hard": -0.1, "metric": "Sharpe"}

    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RF": [0.001, 0.001, 0.001, 0.001],
            "FundA": [0.05, 0.06, -0.05, -0.05],
            "FundB": [0.04, 0.05, 0.04, 0.05],
            "FundC": [0.01, 0.02, 0.05, 0.06],
        }
    )
    res = run_mp(cfg, df)
    assert len(res) == 2
    weights1 = res[1].get("fund_weights", {})
    assert "FundA" not in weights1
    assert "FundB" in weights1


def test_threshold_hold_rank_scores_blended_and_transform() -> None:
    cfg = _make_base_config()
    cfg.portfolio["rank"] = {
        "inclusion_approach": "threshold",
        "score_by": "blended",
        "blended_weights": {"Sharpe": 0.5, "MaxDrawdown": 0.5},
        "threshold": 1.0,
        "transform": "rank",
    }
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.05, 0.05, -0.02, 0.01],
            "FundB": [0.01, -0.02, 0.06, -0.01],
        }
    )
    res = run_mp(cfg, df)
    assert list(res[0].get("fund_weights", {}).keys()) == ["FundA"]
    assert res[0]["fund_weights"]["FundA"] == pytest.approx(0.7)
    ssf = res[0]["selection_score_frame"]
    assert ssf.loc["FundA", "zscore"] == pytest.approx(1.0)
    assert ssf.loc["FundB", "zscore"] == pytest.approx(-1.0)


def test_threshold_hold_custom_weighting() -> None:
    cfg = _make_base_config()
    cfg.portfolio["weighting"] = {"name": "custom"}
    cfg.portfolio["custom_weights"] = {"FundA": 0.65, "FundB": 0.35}
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.05, 0.05, -0.02, 0.01],
            "FundB": [0.01, -0.02, 0.06, -0.01],
        }
    )
    res = run_mp(cfg, df)
    assert len(res) == 2
    w = res[0]["fund_weights"]
    assert w["FundA"] == pytest.approx(0.65)
    assert w["FundB"] == pytest.approx(0.35)


def test_threshold_hold_weight_policy_fallback_equal() -> None:
    cfg = _make_base_config()
    cfg.portfolio["weight_policy"] = {"mode": "drop", "min_assets": 5}
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.05, 0.05, -0.02, 0.01],
            "FundB": [0.01, -0.02, 0.06, -0.01],
        }
    )
    res = run_mp(cfg, df)
    assert len(res) == 2
    assert res[0]["fund_weights"] == {
        "FundA": pytest.approx(0.5),
        "FundB": pytest.approx(0.5),
    }


def test_threshold_hold_risk_weighting_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_base_config()
    cfg.portfolio["weighting"] = {"name": "vol_inverse"}
    cfg.portfolio["constraints"]["max_weight"] = 0.8
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.02, 0.01, 0.02],
            "FundB": [0.01, 0.04, 0.01, 0.04],
        }
    )
    res = run_mp(cfg, df)
    assert len(res) == 2
    weights = res[0]["fund_weights"]
    assert set(weights) == {"FundA", "FundB"}
    assert weights["FundA"] == pytest.approx(0.75)
    assert weights["FundB"] == pytest.approx(0.25)
    assert sum(weights.values()) == pytest.approx(1.0)

    # Missing holding filled with zero (pins line 2870)
    cfg_missing = _make_base_config()
    cfg_missing.portfolio["weighting"] = {"name": "vol_inverse"}
    cfg_missing.portfolio["constraints"]["min_weight"] = 0.0
    cfg_missing.portfolio["constraints"]["max_weight"] = 1.0

    class MockMissingRiskEngine:
        def weight(self, cov: Any) -> pd.Series:
            return pd.Series({"FundA": 1.0})

    orig_resolve = mp_engine._resolve_portfolio_weighting

    def mock_resolve_missing(c: Any) -> Any:
        w, u, r, f, s = orig_resolve(c)
        return w, True, MockMissingRiskEngine(), f, s

    monkeypatch.setattr(mp_engine, "_resolve_portfolio_weighting", mock_resolve_missing)
    res_missing = run_mp(cfg_missing, df)
    assert res_missing[0]["fund_weights"]["FundA"] == pytest.approx(1.0)
    assert "FundB" not in res_missing[0]["fund_weights"]

    # Unnormalized weights normalized by total under cash policy (pins line 2872)
    cfg_unnorm = _make_base_config()
    cfg_unnorm.portfolio["weighting"] = {"name": "vol_inverse"}
    cfg_unnorm.portfolio["weight_policy"] = {"mode": "cash"}
    cfg_unnorm.portfolio["constraints"]["min_weight"] = 0.0
    cfg_unnorm.portfolio["constraints"]["max_weight"] = 1.0

    class MockUnnormRiskEngine:
        def weight(self, cov: Any) -> pd.Series:
            return pd.Series({"FundA": 0.05, "FundB": 0.03})

    def mock_resolve_unnorm(c: Any) -> Any:
        w, u, r, f, s = orig_resolve(c)
        return w, True, MockUnnormRiskEngine(), f, s

    monkeypatch.setattr(mp_engine, "_resolve_portfolio_weighting", mock_resolve_unnorm)
    res_unnorm = run_mp(cfg_unnorm, df)
    weights_unnorm = res_unnorm[0]["fund_weights"]
    assert weights_unnorm["FundA"] == pytest.approx(0.625)
    assert weights_unnorm["FundB"] == pytest.approx(0.375)


def test_threshold_hold_enforce_min_funds_random_mode() -> None:
    cfg = _make_base_config()
    cfg.portfolio["selection_mode"] = "random"
    cfg.portfolio["constraints"]["min_funds"] = 3
    cfg.portfolio["constraints"]["max_funds"] = 4
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "FirmA Fund1": [0.01, 0.02, 0.01, 0.02],
            "FirmA Fund2": [0.01, 0.03, 0.01, 0.03],
            "FirmB Fund1": [0.02, 0.01, 0.02, 0.01],
            "FirmC Fund1": [0.03, 0.02, 0.03, 0.02],
            "FirmD Fund1": [0.01, 0.04, 0.01, 0.04],
        }
    )
    res = run_mp(cfg, df)
    assert len(res) == 2
    weights = res[0]["fund_weights"]
    assert len(weights) >= 3
    firms = [k.split()[0] for k in weights]
    assert len(firms) == len(set(firms))


def test_threshold_hold_rank_blended_normalizes_metric_weights() -> None:
    cfg = _make_base_config()
    cfg.portfolio["rank"] = {
        "inclusion_approach": "threshold",
        "score_by": "blended",
        "blended_weights": {"Sharpe": 1.0, "MaxDrawdown": 3.0},
        "threshold": 0.35,
        "transform": "raw",
    }
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.08, 0.07, 0.06, 0.05],
            "FundB": [0.02, -0.01, 0.03, 0.04],
            "FundC": [-0.02, 0.01, 0.02, 0.03],
        }
    )
    res = run_mp(cfg, df)
    assert list(res[0]["fund_weights"].keys()) == ["FundA", "FundC"]
    assert res[0]["fund_weights"]["FundA"] == pytest.approx(0.5)
    assert res[0]["fund_weights"]["FundC"] == pytest.approx(0.5)


def test_threshold_hold_zero_pipeline_weights_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trend.diagnostics import DiagnosticResult

    original = mp_engine._call_pipeline_with_diag

    def wrap_pipeline(*args: Any, **kwargs: Any) -> DiagnosticResult[dict[str, Any] | None]:
        out = original(*args, **kwargs)
        if out.value is None:
            return out
        payload = dict(out.value)
        manual = [str(x) for x in (kwargs.get("manual_funds") or [])]
        zeroed = {name: 0.0 for name in manual}
        rf_col = kwargs.get("risk_free_column")
        if rf_col:
            zeroed[str(rf_col)] = 0.0
        payload["fund_weights"] = zeroed
        return DiagnosticResult(value=payload, diagnostic=out.diagnostic)

    monkeypatch.setattr(mp_engine, "_call_pipeline_with_diag", wrap_pipeline)

    cfg = _make_base_config()
    cfg.data["allow_risk_free_fallback"] = False
    cfg.data["risk_free_column"] = "RF"
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "RF": [0.001, 0.001, 0.001, 0.001],
            "FundA": [0.05, 0.05, -0.02, 0.01],
            "FundB": [0.01, -0.02, 0.06, -0.01],
        }
    )
    res = run_mp(cfg, df)
    assert len(res) == 2
    assert "RF" not in res[0]["fund_weights"]
    assert res[0]["fund_weights"] == {
        "FundA": pytest.approx(0.5),
        "FundB": pytest.approx(0.5),
    }
    assert res[1]["fund_weights"]["FundA"] == pytest.approx(0.5)
    assert "RF" not in res[1]["fund_weights"]
