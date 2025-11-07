import numpy as np
import pandas as pd
import pytest

import trend_analysis.pipeline as pipeline


def _sample_stats(value: float = 0.1) -> pipeline._Stats:  # type: ignore[name-defined]
    """Helper to build Stats objects with predictable values."""

    return pipeline._Stats(
        cagr=value,
        vol=value + 0.1,
        sharpe=value + 0.2,
        sortino=value + 0.3,
        max_drawdown=-(value + 0.4),
        information_ratio=value + 0.5,
        is_avg_corr=value + 0.6,
        os_avg_corr=value + 0.7,
    )


def test_run_requires_csv_path() -> None:
    with pytest.raises(KeyError):
        pipeline.run({})


@pytest.fixture(name="sample_frame")
def _sample_frame_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=4, freq="ME"),
            "FundA": [0.01, 0.02, 0.0, 0.03],
            "FundB": [0.0, -0.01, 0.02, 0.01],
        }
    )


@pytest.fixture(name="sample_split")
def _sample_split_fixture() -> dict[str, str]:
    return {
        "in_start": "2020-01",
        "in_end": "2020-02",
        "out_start": "2020-03",
        "out_end": "2020-04",
    }


@pytest.fixture(name="base_config")
def _base_config_fixture() -> dict[str, object]:
    return {
        "data": {"csv_path": "dummy.csv"},
        "sample_split": {},
        "preprocessing": {},
        "run": {},
        "portfolio": {},
        "vol_adjust": {},
    }


def test_run_converts_stats_payload_to_frame(
    monkeypatch: pytest.MonkeyPatch,
    sample_frame: pd.DataFrame,
    sample_split: dict[str, str],
    base_config: dict[str, object],
) -> None:
    monkeypatch.setattr(pipeline, "load_csv", lambda *_, **__: sample_frame)
    monkeypatch.setattr(
        pipeline, "_resolve_sample_split", lambda *_args, **_kwargs: sample_split
    )
    monkeypatch.setattr(
        pipeline, "_build_trend_spec", lambda *_args, **_kwargs: object()
    )

    stats_payload = {
        "FundA": _sample_stats(0.1),
        "FundB": _sample_stats(0.2),
    }
    benchmark_ir = {
        "custom": {"FundA": 1.23, "FundB": 0.98, "equal_weight": 0.0},
        "user_weight": {"FundA": 0.5},
    }

    def fake_run_analysis(*args, **kwargs):
        assert args[0] is sample_frame
        assert kwargs["signal_spec"] is not None
        return {
            "out_sample_stats": stats_payload,
            "benchmark_ir": benchmark_ir,
            "risk_diagnostics": {},
            "fund_weights": {},
            "in_sample_stats": {},
        }

    monkeypatch.setattr(pipeline, "_run_analysis", fake_run_analysis)

    result = pipeline.run(base_config)
    assert list(result.index) == ["FundA", "FundB"]
    assert set(result.columns) >= {
        "cagr",
        "vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "information_ratio",
        "ir_custom",
    }
    # Synthetic weight aggregates still surface as dedicated columns.
    assert result.loc["FundA", "ir_user_weight"] == pytest.approx(0.5)
    assert result.loc["FundA", "ir_custom"] == pytest.approx(1.23)


def test_run_returns_empty_frame_when_analysis_none(
    monkeypatch: pytest.MonkeyPatch,
    sample_frame: pd.DataFrame,
    sample_split: dict[str, str],
    base_config: dict[str, object],
) -> None:
    monkeypatch.setattr(pipeline, "load_csv", lambda *_, **__: sample_frame)
    monkeypatch.setattr(
        pipeline, "_resolve_sample_split", lambda *_args, **_kwargs: sample_split
    )
    monkeypatch.setattr(
        pipeline, "_build_trend_spec", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(pipeline, "_run_analysis", lambda *_, **__: None)

    result = pipeline.run(base_config)
    assert result.empty


def test_run_full_propagates_analysis_payload(
    monkeypatch: pytest.MonkeyPatch,
    sample_frame: pd.DataFrame,
    sample_split: dict[str, str],
    base_config: dict[str, object],
) -> None:
    monkeypatch.setattr(pipeline, "load_csv", lambda *_, **__: sample_frame)
    monkeypatch.setattr(
        pipeline, "_resolve_sample_split", lambda *_args, **_kwargs: sample_split
    )
    monkeypatch.setattr(
        pipeline, "_build_trend_spec", lambda *_args, **_kwargs: object()
    )

    payload = {
        "out_sample_stats": {"FundA": _sample_stats(0.4)},
        "benchmark_ir": {},
        "risk_diagnostics": {"entries": 1},
    }
    monkeypatch.setattr(pipeline, "_run_analysis", lambda *_, **__: payload)

    result = pipeline.run_full(base_config)
    assert result is payload


def test_run_full_returns_empty_when_analysis_none(
    monkeypatch: pytest.MonkeyPatch,
    sample_frame: pd.DataFrame,
    sample_split: dict[str, str],
    base_config: dict[str, object],
) -> None:
    monkeypatch.setattr(pipeline, "load_csv", lambda *_, **__: sample_frame)
    monkeypatch.setattr(
        pipeline, "_resolve_sample_split", lambda *_args, **_kwargs: sample_split
    )
    monkeypatch.setattr(
        pipeline, "_build_trend_spec", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(pipeline, "_run_analysis", lambda *_, **__: None)

    result = pipeline.run_full(base_config)
    assert result == {}


def test_empty_run_full_result_template() -> None:
    payload = pipeline._empty_run_full_result()
    assert set(payload.keys()) == {
        "out_sample_stats",
        "in_sample_stats",
        "benchmark_ir",
        "risk_diagnostics",
        "fund_weights",
    }


def test_compute_stats_includes_optional_avg_corr() -> None:
    data = pd.DataFrame(
        {
            "FundA": [0.01, 0.02, 0.0, -0.01],
            "FundB": [0.0, 0.01, 0.02, 0.03],
        }
    )
    rf = pd.Series([0.0, 0.001, 0.002, 0.0])
    stats = pipeline._compute_stats(
        data,
        rf,
        in_sample_avg_corr={"FundA": 0.5},
        out_sample_avg_corr={"FundB": 0.25},
    )
    assert stats["FundA"].is_avg_corr == 0.5
    assert stats["FundB"].os_avg_corr == 0.25


def test_calc_portfolio_returns_scales_weights(sample_frame: pd.DataFrame) -> None:
    weights = np.array([0.6, 0.4])
    portfolio = pipeline.calc_portfolio_returns(
        weights, sample_frame[["FundA", "FundB"]]
    )
    assert isinstance(portfolio, pd.Series)
    expected = (sample_frame[["FundA", "FundB"]] * weights).sum(axis=1)
    pd.testing.assert_series_equal(portfolio, expected)
