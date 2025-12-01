from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from trend.diagnostics import DiagnosticResult
from trend_analysis import api
from trend_analysis.config import Config
from trend_analysis.config.model import validate_trend_config
from trend_analysis.diagnostics import PipelineReasonCode, pipeline_failure
from trend_analysis.multi_period import engine as mp_engine
from trend_analysis.pipeline import _resolve_risk_free_column
from trend_analysis.util.risk_free import resolve_risk_free_settings


@dataclass
class _MultiPeriodDummyConfig:
    multi_period: Dict[str, Any] = field(
        default_factory=lambda: {
            "frequency": "M",
            "in_sample_len": 1,
            "out_sample_len": 1,
            "start": "2020-01",
            "end": "2020-03",
        }
    )
    data: Dict[str, Any] = field(default_factory=dict)
    portfolio: Dict[str, Any] = field(
        default_factory=lambda: {
            "policy": "standard",
            "selection_mode": "all",
            "random_n": 2,
            "custom_weights": None,
            "rank": {},
            "manual_list": None,
            "indices_list": None,
        }
    )
    vol_adjust: Dict[str, Any] = field(default_factory=lambda: {"target_vol": 1.0})
    benchmarks: List[Any] = field(default_factory=list)
    run: Dict[str, Any] = field(default_factory=lambda: {"monthly_cost": 0.0})
    seed: int = 123

    def model_dump(self) -> Dict[str, Any]:
        return {
            "multi_period": self.multi_period,
            "portfolio": self.portfolio,
            "vol_adjust": self.vol_adjust,
        }


def _make_single_config(
    allow_risk_free_fallback: bool | None, risk_free_column: str | None
) -> Config:
    data_section: Dict[str, Any] = {"date_column": "Date", "frequency": "M"}
    if risk_free_column is not None:
        data_section["risk_free_column"] = risk_free_column
    if allow_risk_free_fallback is not None:
        data_section["allow_risk_free_fallback"] = allow_risk_free_fallback

    return Config(
        version="1",
        data=data_section,
        preprocessing={},
        vol_adjust={"target_vol": 1.0},
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-01",
            "out_start": "2020-02",
            "out_end": "2020-02",
        },
        portfolio={},
        metrics={},
        export={},
        run={},
    )


def _make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-31", periods=2, freq="ME"),
            "AssetA": [0.01, 0.02],
        }
    )


@pytest.mark.parametrize("risk_free_column", [None, "RF"])
@pytest.mark.parametrize(
    "allow_value, expected",
    [
        (None, False),
        (True, True),
        (False, False),
    ],
)
def test_entry_points_resolve_risk_free_settings_consistently(
    risk_free_column: str | None,
    allow_value: bool | None,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _make_frame()
    cfg = _make_single_config(allow_value, risk_free_column)
    expected_allow = False if risk_free_column else expected

    single_invocations: list[dict[str, Any]] = []

    def fake_single_run(*_: Any, **kwargs: Any):
        single_invocations.append(kwargs)
        return pipeline_failure(PipelineReasonCode.NO_FUNDS_SELECTED)

    monkeypatch.setattr(api, "_run_analysis", fake_single_run)
    api.run_simulation(cfg, frame)

    assert single_invocations
    single_kwargs = single_invocations[-1]
    assert single_kwargs["risk_free_column"] == risk_free_column
    assert single_kwargs["allow_risk_free_fallback"] is expected_allow

    multi_cfg = _MultiPeriodDummyConfig()
    multi_cfg.data = {"date_column": "Date", "frequency": "M"}
    if risk_free_column is not None:
        multi_cfg.data["risk_free_column"] = risk_free_column
    if allow_value is not None:
        multi_cfg.data["allow_risk_free_fallback"] = allow_value

    multi_invocations: list[dict[str, Any]] = []

    def fake_multi_run(
        *_: Any, **kwargs: Any
    ) -> DiagnosticResult[dict[str, Any] | None]:
        multi_invocations.append(kwargs)
        return pipeline_failure(PipelineReasonCode.NO_FUNDS_SELECTED)

    monkeypatch.setattr(mp_engine, "_run_analysis", fake_multi_run)
    mp_engine.run(multi_cfg, frame)

    assert multi_invocations
    multi_kwargs = multi_invocations[-1]
    assert multi_kwargs["risk_free_column"] == risk_free_column
    assert multi_kwargs["allow_risk_free_fallback"] is expected_allow


def test_missing_risk_free_requires_explicit_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _make_frame()
    cfg = _make_single_config(None, None)

    def _run_with_risk_free_check(*_: Any, **kwargs: Any):
        _resolve_risk_free_column(
            frame,
            date_col="Date",
            indices_list=None,
            risk_free_column=kwargs.get("risk_free_column"),
            allow_risk_free_fallback=kwargs.get("allow_risk_free_fallback"),
        )
        return pipeline_failure(PipelineReasonCode.NO_FUNDS_SELECTED)

    monkeypatch.setattr(api, "_run_analysis", _run_with_risk_free_check)

    with pytest.raises(ValueError) as single_err:
        api.run_simulation(cfg, frame)

    single_message = str(single_err.value)

    multi_cfg = _MultiPeriodDummyConfig()
    multi_cfg.data = {"date_column": "Date", "frequency": "M"}

    monkeypatch.setattr(mp_engine, "_run_analysis", _run_with_risk_free_check)

    with pytest.raises(ValueError) as multi_err:
        mp_engine.run(multi_cfg, frame)

    assert str(multi_err.value) == single_message
    assert "allow_risk_free_fallback" in single_message


@pytest.mark.parametrize("risk_free_column", [None, "RF"])
@pytest.mark.parametrize(
    "allow_value, expected", [(None, False), (True, True), (False, False)]
)
def test_trend_config_validation_resolves_defaults(
    allow_value: bool | None,
    risk_free_column: str | None,
    expected: bool,
    tmp_path: Path,
) -> None:
    data_section: dict[str, Any] = {"date_column": "Date", "frequency": "M"}
    if risk_free_column is not None:
        data_section["risk_free_column"] = risk_free_column
    if allow_value is not None:
        data_section["allow_risk_free_fallback"] = allow_value

    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("Date,AssetA\n2020-01-31,0.0\n")
    data_section["csv_path"] = csv_path

    cfg = validate_trend_config(
        {
            "data": data_section,
            "portfolio": {
                "rebalance_calendar": "NYSE",
                "max_turnover": 0.5,
                "transaction_cost_bps": 0.0,
            },
            "vol_adjust": {"target_vol": 1.0},
        },
        base_path=tmp_path,
    )

    resolved_col, resolved_allow = resolve_risk_free_settings(cfg.data.model_dump())
    assert resolved_col == risk_free_column
    expected_allow = False if risk_free_column else expected
    assert resolved_allow is expected_allow
