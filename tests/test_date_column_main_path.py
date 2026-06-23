from types import SimpleNamespace

import pandas as pd
import pytest
from trend.diagnostics import DiagnosticResult

from trend_analysis import pipeline_entrypoints
from trend_analysis.data import load_csv
from trend_analysis.io.market_data import MarketDataValidationError
from trend_analysis.multi_period import engine as multi_period_engine
from trend_analysis.multi_period import loaders as multi_period_loaders


class DummyResult(SimpleNamespace):
    metrics: pd.DataFrame
    details: dict


def _section_get(section, key, default=None):
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


def _cfg_section(cfg, key):
    return _section_get(cfg, key, {})


def _pipeline_bindings(captured: dict[str, object]) -> pipeline_entrypoints.ConfigBindings:
    def fake_load_csv(path, *, errors="raise", date_column="Date", **kwargs):
        captured.update(
            {
                "path": path,
                "errors": errors,
                "date_column": date_column,
                "kwargs": kwargs,
            }
        )
        return pd.DataFrame({"Date": pd.to_datetime(["2024-01-31"]), "ManagerA": [0.01]})

    def fake_invoke_analysis(*args, **kwargs):
        del args, kwargs
        return DiagnosticResult.success(
            {
                "out_sample_stats": {"ManagerA": SimpleNamespace(total_return=0.01)},
                "benchmark_ir": {},
            }
        )

    return pipeline_entrypoints.ConfigBindings(
        load_csv=fake_load_csv,
        attach_calendar_settings=lambda *_args, **_kwargs: None,
        unwrap_cfg=lambda cfg: cfg,
        cfg_section=_cfg_section,
        section_get=_section_get,
        cfg_value=lambda cfg, key, default=None: _section_get(cfg, key, default),
        resolve_sample_split=lambda _df, split: split,
        policy_from_config=lambda _missing: (None, None),
        build_trend_spec=lambda *_args, **_kwargs: None,
        resolve_target_vol=lambda _vol_adjust: None,
        invoke_analysis_with_diag=fake_invoke_analysis,
        weight_engine_params_from_robustness=lambda *_args, **_kwargs: None,
        RiskStatsConfig=lambda **kwargs: SimpleNamespace(**kwargs),
    )


def test_load_csv_honors_configured_date_column(tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text(
        "Timestamp,ManagerA,ManagerB\n" "2024-01-31,0.01,0.02\n" "2024-02-29,0.03,0.04\n"
    )

    frame = load_csv(str(csv_path), errors="raise", date_column="Timestamp")

    assert frame is not None
    assert list(frame.columns) == ["Date", "ManagerA", "ManagerB"]
    assert pd.api.types.is_datetime64_any_dtype(frame["Date"])


def test_load_csv_still_accepts_standard_date_column(tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("Date,ManagerA\n2024-01-31,0.01\n2024-02-29,0.03\n")

    frame = load_csv(str(csv_path), errors="raise")

    assert frame is not None
    assert list(frame.columns) == ["Date", "ManagerA"]


def test_load_csv_rejects_missing_configured_date_column(tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("Date,ManagerA\n" "2024-01-31,0.01\n" "2024-02-29,0.03\n")

    with pytest.raises(MarketDataValidationError, match="Timestamp"):
        load_csv(str(csv_path), errors="raise", date_column="Timestamp")


def test_accepts_keyword_is_conservative_when_signature_fails():
    marker = object()

    assert pipeline_entrypoints._accepts_keyword(marker, "date_column") is False
    assert multi_period_loaders._accepts_keyword(marker, "date_column") is False
    assert multi_period_engine._accepts_keyword(marker, "date_column") is False


def test_pipeline_entrypoint_passes_configured_date_column(tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("Timestamp,ManagerA\n2024-01-31,0.01\n")
    cfg = SimpleNamespace(
        data={"csv_path": str(csv_path), "date_column": "Timestamp"},
        sample_split={
            "in_start": "2024-01-01",
            "in_end": "2024-01-31",
            "out_start": "2024-02-01",
            "out_end": "2024-02-29",
        },
        export={"directory": str(tmp_path), "formats": ["json"], "filename": "report"},
    )

    captured: dict[str, object] = {}

    result = pipeline_entrypoints.run_from_config(cfg, bindings=_pipeline_bindings(captured))

    assert not result.empty
    assert captured["path"] == str(csv_path)
    assert captured["errors"] == "raise"
    assert captured["date_column"] == "Timestamp"


def test_pipeline_entrypoint_without_date_column_keeps_default(tmp_path):
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("Date,ManagerA\n2024-01-31,0.01\n")
    cfg = SimpleNamespace(
        data={"csv_path": str(csv_path)},
        sample_split={
            "in_start": "2024-01-01",
            "in_end": "2024-01-31",
            "out_start": "2024-02-01",
            "out_end": "2024-02-29",
        },
        export={"directory": str(tmp_path), "formats": ["json"], "filename": "report"},
    )

    captured: dict[str, object] = {}

    result = pipeline_entrypoints.run_from_config(cfg, bindings=_pipeline_bindings(captured))

    assert not result.empty
    assert captured["date_column"] == "Date"
