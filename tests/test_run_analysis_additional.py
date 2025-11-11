"""Extra coverage for ``trend_analysis.run_analysis`` CLI entry point."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis import run_analysis as run_analysis_mod
from trend_analysis.constants import DEFAULT_OUTPUT_DIRECTORY


class DummyResult:
    def __init__(self, metrics: pd.DataFrame, details: dict[str, object]):
        self.metrics = metrics
        self.details = details


@pytest.fixture()
def basic_config() -> SimpleNamespace:
    return SimpleNamespace(
        data={"csv_path": "input.csv"},
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-06",
            "out_start": "2020-07",
            "out_end": "2020-12",
        },
        export={"directory": "out", "formats": ["json"], "filename": "analysis"},
    )


def test_main_requires_csv_path(
    monkeypatch: pytest.MonkeyPatch, basic_config: SimpleNamespace
) -> None:
    cfg = SimpleNamespace(data={}, sample_split={}, export={})
    monkeypatch.setattr(run_analysis_mod, "load", lambda path: cfg)

    with pytest.raises(KeyError):
        run_analysis_mod.main(["-c", "config.yml"])


def test_main_raises_file_not_found_when_load_returns_none(
    monkeypatch: pytest.MonkeyPatch, basic_config: SimpleNamespace
) -> None:
    monkeypatch.setattr(run_analysis_mod, "load", lambda path: basic_config)
    monkeypatch.setattr(run_analysis_mod, "load_csv", lambda path, **_: None)

    with pytest.raises(FileNotFoundError):
        run_analysis_mod.main(["-c", "config.yml"])


def test_main_handles_nan_policy_without_errors_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SimpleNamespace(
        data={
            "csv_path": Path("input.csv"),
            "nan_policy": {"FundA": "ffill"},
            "nan_limit": {"FundA": 3},
        },
        sample_split={},
        export={"directory": "out", "formats": ["json"]},
    )

    captured: dict[str, object] = {}

    def fake_load_csv(path: str, *, nan_policy=None, nan_limit=None) -> pd.DataFrame:
        captured["nan_policy"] = nan_policy
        captured["nan_limit"] = nan_limit
        return pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-31", periods=2, freq="ME"),
                "FundA": [0.1, 0.2],
            }
        )

    result = DummyResult(
        metrics=pd.DataFrame({"metric": [1.0]}),
        details={"metrics": pd.DataFrame()},
    )

    with monkeypatch.context() as mp:
        mp.setattr(run_analysis_mod, "load", lambda path: cfg)
        mp.setattr(run_analysis_mod, "load_csv", fake_load_csv)
        mp.setattr(run_analysis_mod.api, "run_simulation", lambda cfg, df: result)

        rc = run_analysis_mod.main(["-c", "config.yml", "--detailed"])

    assert rc == 0
    assert captured["nan_policy"] == cfg.data["nan_policy"]
    assert captured["nan_limit"] == cfg.data["nan_limit"]
    assert "errors" not in captured


def test_main_assigns_default_output_when_missing_export_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SimpleNamespace(
        data={"csv_path": "input.csv", "missing_policy": {"FundA": "drop"}},
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-06",
            "out_start": "2020-07",
            "out_end": "2020-12",
        },
        export={},
    )

    frame = pd.DataFrame({"metric": [1.0]})
    details = {
        "summary": "ok",
        "performance_by_regime": pd.DataFrame(),
        "regime_notes": ["note"],
    }
    result = DummyResult(metrics=frame, details=details)

    exported: dict[str, object] = {}

    def fake_load_csv(path: str, **kwargs: object) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-31", periods=3, freq="ME"),
                "FundA": [0.1, 0.2, 0.3],
            }
        )

    with monkeypatch.context() as mp:
        mp.setattr(run_analysis_mod, "load", lambda path: cfg)
        mp.setattr(run_analysis_mod, "load_csv", fake_load_csv)
        mp.setattr(run_analysis_mod.api, "run_simulation", lambda cfg, df: result)
        mp.setattr(
            run_analysis_mod.export, "format_summary_text", lambda *_, **__: "summary"
        )
        mp.setattr(
            run_analysis_mod.export,
            "make_summary_formatter",
            lambda *_, **__: lambda df: None,
        )
        mp.setattr(
            run_analysis_mod.export,
            "summary_frame_from_result",
            lambda details: pd.DataFrame({"metric": [1.0]}),
        )
        mp.setattr(
            run_analysis_mod.export,
            "export_to_excel",
            lambda data, path, **kwargs: exported.update(
                {
                    "excel_path": Path(path),
                    "keys": tuple(sorted(data)),
                }
            ),
        )
        mp.setattr(
            run_analysis_mod.export,
            "export_data",
            lambda *args, **kwargs: exported.setdefault("data_calls", []).append(args),
        )

        rc = run_analysis_mod.main(["-c", "config.yml"])

    assert rc == 0
    assert exported["excel_path"].parent == Path(DEFAULT_OUTPUT_DIRECTORY)
    assert exported["excel_path"].name == "analysis.xlsx"
    assert "metrics" in exported["keys"]
