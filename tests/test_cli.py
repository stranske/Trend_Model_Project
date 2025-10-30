from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from trend_analysis import cli
from trend_analysis.api import RunResult
from trend_analysis.constants import DEFAULT_OUTPUT_DIRECTORY, DEFAULT_OUTPUT_FORMATS
from trend_analysis.io.market_data import MarketDataValidationError

cache_first = {
    "entries": 1,
    "hits": 2,
    "misses": 3,
    "incremental_updates": 4,
}

cache_second = {
    "entries": 5.0,
    "hits": 6.0,
    "misses": 7.0,
    "incremental_updates": 8.0,
}


# Helper class for tests
class TruthySeries:
    def __init__(self, series: pd.Series):
        self.series = series

    def __bool__(self) -> bool:
        return True

    def __getattr__(self, name: str):
        return getattr(self.series, name)


def _write_cfg(path: Path, version: str, *, csv_path: Path) -> None:
    try:
        csv_value = str(csv_path.relative_to(path.parent))
    except ValueError:
        csv_value = str(csv_path)

    config = {
        "version": version,
        "data": {
            "csv_path": csv_value,
            "date_column": "Date",
            "frequency": "M",
        },
        "preprocessing": {},
        "vol_adjust": {"target_vol": 0.15},
        "sample_split": {},
        "portfolio": {
            "selection_mode": "all",
            "rebalance_calendar": "NYSE",
            "max_turnover": 0.25,
            "transaction_cost_bps": 10,
        },
        "metrics": {},
        "export": {},
        "run": {},
    }

    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_cli_run_with_preset_applies_signals(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yml"
    csv = tmp_path / "data.csv"
    csv.write_text("Date,A\n2020-01-31,0.0\n")
    _write_cfg(cfg, "1", csv_path=csv)

    captured: dict[str, object] = {}

    def fake_run_simulation(cfg_obj, df):
        captured["signals"] = getattr(cfg_obj, "signals", {})
        captured["vol_adjust"] = getattr(cfg_obj, "vol_adjust", {})
        captured["run"] = getattr(cfg_obj, "run", {})
        return RunResult(pd.DataFrame(), {"out_sample_stats": {}}, 42, {})

    original_load_config = cli.load_config

    def fake_load_config(path):
        cfg_obj = original_load_config(path)
        setattr(
            cfg_obj,
            "sample_split",
            {
                "in_start": "2020-01",
                "in_end": "2020-02",
                "out_start": "2020-03",
                "out_end": "2020-04",
            },
        )
        return cfg_obj

    monkeypatch.setattr(cli, "load_config", fake_load_config)

    frame = pd.DataFrame({"Date": pd.to_datetime(["2020-01-31"]), "A": [0.0]})
    monkeypatch.setattr(
        cli,
        "load_market_data_csv",
        lambda path: SimpleNamespace(frame=frame),
    )
    monkeypatch.setattr(cli, "run_simulation", fake_run_simulation)
    monkeypatch.setattr(cli.export, "format_summary_text", lambda *a, **k: "")
    monkeypatch.setattr(cli.export, "export_to_excel", lambda *a, **k: None)
    monkeypatch.setattr(cli.export, "export_data", lambda *a, **k: None)
    monkeypatch.setattr(cli.run_logging, "init_run_logger", lambda *a, **k: None)

    rc = cli.main(["run", "-c", str(cfg), "-i", str(csv), "--preset", "conservative"])

    assert rc == 0
    signals = captured["signals"]
    assert isinstance(signals, dict)
    assert signals["window"] == 126
    assert signals["vol_adjust"] is True
    assert pytest.approx(signals["vol_target"], rel=1e-6) == 0.08

    vol_adjust = captured["vol_adjust"]
    assert isinstance(vol_adjust, dict)
    assert vol_adjust["window"]["length"] == 126

    run_section = captured["run"]
    assert isinstance(run_section, dict)
    assert run_section.get("trend_preset") == "conservative"


def test_cli_version_custom(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yml"
    csv = tmp_path / "data.csv"
    csv.write_text("Date,RF\n2020-01-31,0.0\n2020-02-29,0.1\n")
    _write_cfg(cfg, "1.2.3", csv_path=csv)

    captured: dict[str, str] = {}

    def fake_run(cfg):
        captured["version"] = cfg.version
        return pd.DataFrame()

    monkeypatch.setattr(cli.pipeline, "run", fake_run)
    monkeypatch.setattr(cli.pipeline, "run_full", lambda cfg: {"dummy": 1})
    monkeypatch.setattr(cli.export, "format_summary_text", lambda *a, **k: "")
    monkeypatch.setattr(cli.export, "export_to_excel", lambda *a, **k: None)
    monkeypatch.setattr(cli.export, "export_data", lambda *a, **k: None)

    rc = cli.main(["run", "-c", str(cfg), "-i", str(csv)])
    assert rc == 0
    assert captured["version"] == "1.2.3"


def test_cli_default_json(tmp_path, capsys, monkeypatch):
    cfg = tmp_path / "cfg.yml"
    csv = tmp_path / "data.csv"
    csv.write_text("Date,RF\n2020-01-31,0.0\n2020-02-29,0.1\n")
    _write_cfg(cfg, "1", csv_path=csv)

    monkeypatch.setattr(cli.pipeline, "run", lambda cfg: pd.DataFrame())
    monkeypatch.setattr(cli.pipeline, "run_full", lambda cfg: None)

    rc = cli.main(["run", "-c", str(cfg), "-i", str(csv)])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out == "No results"


def test_cli_validation_error(monkeypatch, capsys):
    config = SimpleNamespace(
        seed=1,
        sample_split={},
        export={"directory": "outputs", "formats": ["csv"]},
        run={},
        vol_adjust={},
        metrics={},
        portfolio={},
        benchmarks={},
    )

    monkeypatch.setattr(cli, "load_config", lambda path: config)

    def raise_validation(path: str):
        raise MarketDataValidationError("Data validation failed:\n• unsorted index")

    monkeypatch.setattr(cli, "load_market_data_csv", raise_validation)

    rc = cli.main(["run", "-c", "cfg.yml", "-i", "input.csv"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "unsorted index" in captured.err


def test_cli_run_legacy_bundle_and_exports(tmp_path, capsys, monkeypatch):
    config = SimpleNamespace(
        seed=123,
        sample_split={"in_start": "2020-01-01"},
        export={
            "directory": str(tmp_path),
            "formats": ["excel", "json"],
            "filename": "report",
        },
        run={},
        vol_adjust={},
        metrics={},
        portfolio={},
        benchmarks={},
    )

    metrics_df = pd.DataFrame({"Sharpe": [1.23]})
    results_payload = {"summary": "ok"}

    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "load_market_data_csv",
        lambda path: SimpleNamespace(
            frame=pd.DataFrame({"Date": pd.to_datetime(["2020-01-31"]), "A": [0.0]})
        ),
    )
    monkeypatch.setattr(cli.pipeline, "run", lambda cfg: metrics_df)
    monkeypatch.setattr(cli.pipeline, "run_full", lambda cfg: results_payload)
    monkeypatch.setattr(cli.export, "format_summary_text", lambda *a, **k: "summary")

    formatter_calls: list[tuple[dict, tuple[str, str, str, str]]] = []
    monkeypatch.setattr(
        cli.export,
        "make_summary_formatter",
        lambda res, *periods: (
            formatter_calls.append((res, tuple(periods))) or object()
        ),
    )

    excel_calls: list[tuple[dict, Path, object]] = []
    monkeypatch.setattr(
        cli.export,
        "export_to_excel",
        lambda data, path, default_sheet_formatter: excel_calls.append(
            (data, Path(path), default_sheet_formatter)
        ),
    )

    data_calls: list[tuple[dict, Path, tuple[str, ...]]] = []
    monkeypatch.setattr(
        cli.export,
        "export_data",
        lambda data, path, formats: data_calls.append(
            (data, Path(path), tuple(formats))
        ),
    )

    bundle_calls: dict[str, object] = {}

    def fake_export_bundle(rr, path):
        bundle_calls["rr"] = rr
        bundle_calls["path"] = path

    monkeypatch.setattr(
        "trend_analysis.export.bundle.export_bundle", fake_export_bundle
    )

    rc = cli.main(
        [
            "run",
            "-c",
            "config.yml",
            "-i",
            "input.csv",
            "--bundle",
            str(tmp_path),
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "Bundle written" in out
    assert formatter_calls == [
        (results_payload, ("2020-01-01", "None", "None", "None"))
    ]
    assert excel_calls and excel_calls[0][1] == tmp_path / "report.xlsx"
    excel_payload, _, _ = excel_calls[0]
    assert excel_payload["metrics"] is metrics_df

    assert len(data_calls) == 1
    data_payload, data_path, formats = data_calls[0]
    assert data_payload is excel_payload
    assert data_path == tmp_path / "report"
    assert formats == ("json",)

    assert bundle_calls["path"] == tmp_path / "analysis_bundle.zip"
    rr = bundle_calls["rr"]
    assert isinstance(rr, RunResult)
    assert rr.metrics is metrics_df
    assert rr.details is results_payload
    assert rr.seed == 123
    assert rr.config["export"]["filename"] == "report"
    assert rr.input_path == Path("input.csv")


def test_cli_run_modern_bundle_attaches_payload(tmp_path, capsys, monkeypatch):
    out_dir = tmp_path / "exports"
    out_dir.mkdir()

    split = {
        "in_start": "2020-01-01",
        "in_end": "2020-12-31",
        "out_start": "2021-01-01",
        "out_end": "2021-12-31",
    }

    config = SimpleNamespace(
        seed=5,
        sample_split=split,
        export={"directory": str(out_dir), "formats": ["json"], "filename": "custom"},
        run={},
        vol_adjust={},
        metrics={},
        portfolio={},
        benchmarks={},
    )

    metrics_df = pd.DataFrame({"Return": [0.05]})
    portfolio_series = pd.Series(
        [0.1, 0.2], index=pd.Index(["2020-01", "2020-02"]), name="user"
    )

    class TruthySeries:
        def __init__(self, series: pd.Series):
            self.series = series

        def __bool__(self) -> bool:
            return True

        def equals(self, other: pd.Series) -> bool:  # pragma: no cover - passthrough
            return self.series.equals(other)

        def __getattr__(self, name):
            return getattr(self.series, name)

    details = {
        "portfolio_user_weight": 0,
        "portfolio_equal_weight": TruthySeries(portfolio_series),
        "benchmarks": {
            "bench": pd.Series(
                [0.3, 0.4], index=pd.Index(["2020-01", "2020-02"]), name="bench"
            )
        },
        "weights_user_weight": pd.DataFrame({"w": [1.0]}, index=["fund"]),
    }
    run_result = RunResult(metrics_df, details, 999, {"python": "3"})

    seed_seen: dict[str, int] = {}

    def fake_run_simulation(cfg, df):
        seed_seen["seed"] = getattr(cfg, "seed")
        return run_result

    monkeypatch.setenv("TREND_SEED", "456")
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "load_market_data_csv",
        lambda path: SimpleNamespace(
            frame=pd.DataFrame({"Date": pd.to_datetime(["2020-01-31"]), "A": [0.0]})
        ),
    )
    monkeypatch.setattr(cli, "run_simulation", fake_run_simulation)
    monkeypatch.setattr(cli.export, "format_summary_text", lambda *a, **k: "summary")

    monkeypatch.setattr(cli.export, "export_data", lambda *a, **k: None)
    monkeypatch.setattr(
        cli.export,
        "export_to_excel",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("unexpected excel export")
        ),
    )

    bundle_calls: dict[str, object] = {}

    def fake_export_bundle(rr, path):
        bundle_calls["rr"] = rr
        bundle_calls["path"] = path

    monkeypatch.setattr(
        "trend_analysis.export.bundle.export_bundle", fake_export_bundle
    )

    monkeypatch.chdir(tmp_path)
    rc = cli.main(
        [
            "run",
            "-c",
            "cfg.yml",
            "-i",
            "input.csv",
            "--seed",
            "789",
            "--bundle",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "summary" in out
    assert seed_seen["seed"] == 789
    assert config.seed == 789
    assert run_result.portfolio.equals(portfolio_series)
    assert run_result.benchmark.equals(details["benchmarks"]["bench"])
    assert run_result.weights.equals(details["weights_user_weight"])
    assert bundle_calls["rr"] is run_result
    assert bundle_calls["path"] == Path("analysis_bundle.zip")


def test_cli_run_env_seed_and_default_exports(tmp_path, capsys, monkeypatch):
    out_dir = Path(DEFAULT_OUTPUT_DIRECTORY)
    split = {
        "in_start": "2019-01-01",
        "in_end": "2019-12-31",
        "out_start": "2020-01-01",
        "out_end": "2020-12-31",
    }

    config = SimpleNamespace(
        seed=11,
        sample_split=split,
        export={},
        run={},
        vol_adjust={},
        metrics={},
        portfolio={},
        benchmarks={},
    )

    metrics_df = pd.DataFrame({"Sharpe": [1.5]})
    run_result = RunResult(metrics_df, {"summary": "great"}, 222, {"python": "3.12"})

    seen: dict[str, int] = {}

    def fake_run_simulation(cfg, df):
        seen["seed"] = getattr(cfg, "seed")
        return run_result

    monkeypatch.setenv("TREND_SEED", "314")
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "load_market_data_csv",
        lambda path: SimpleNamespace(
            frame=pd.DataFrame({"Date": pd.to_datetime(["2019-01-31"]), "A": [0.0]})
        ),
    )
    monkeypatch.setattr(cli, "run_simulation", fake_run_simulation)
    monkeypatch.setattr(cli.export, "format_summary_text", lambda *a, **k: "summary")

    excel_calls: list[tuple[dict, Path, object]] = []
    monkeypatch.setattr(
        cli.export,
        "export_to_excel",
        lambda data, path, default_sheet_formatter: excel_calls.append(
            (data, Path(path), default_sheet_formatter)
        ),
    )

    data_calls: list[tuple[dict, Path, tuple[str, ...]]] = []
    monkeypatch.setattr(
        cli.export,
        "export_data",
        lambda data, path, formats: data_calls.append(
            (data, Path(path), tuple(formats))
        ),
    )

    monkeypatch.setattr(
        cli.export,
        "make_summary_formatter",
        lambda *a, **k: object(),
    )

    rc = cli.main(["run", "-c", "cfg.yml", "-i", "input.csv"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "summary" in out
    assert seen["seed"] == 314
    assert config.seed == 314
    assert excel_calls and excel_calls[0][1] == out_dir / "analysis.xlsx"
    data_payload, data_path, formats = data_calls[0]
    assert data_payload is excel_calls[0][0]
    assert data_path == out_dir / "analysis"
    assert formats == tuple(DEFAULT_OUTPUT_FORMATS)


def test_maybe_log_step_respects_flag(monkeypatch):
    """Structured logging helper should be a no-op when disabled."""

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_log_step(run_id: str, event: str, message: str, **fields: object) -> None:
        calls.append(((run_id, event, message), dict(fields)))

    monkeypatch.setattr(cli, "_log_step", fake_log_step)

    cli.maybe_log_step(False, "rid", "cache", "skipped", extra=1)
    assert calls == []

    cli.maybe_log_step(True, "rid", "cache", "processed", entries=2)
    assert calls == [(("rid", "cache", "processed"), {"entries": 2})]


def test_cli_run_uses_env_seed_and_populates_run_result(tmp_path, capsys, monkeypatch):
    out_dir = tmp_path / "exports"
    out_dir.mkdir()

    config = SimpleNamespace(
        seed=None,
        sample_split={
            "in_start": "2020-01-01",
            "in_end": "2020-12-31",
            "out_start": "2021-01-01",
            "out_end": "2021-12-31",
        },
        export={
            "directory": str(out_dir),
            "formats": ["excel", "json"],
            "filename": "analysis",
        },
        run={},
        vol_adjust={},
        metrics={},
        portfolio={},
        benchmarks={},
    )

    metrics_df = pd.DataFrame({"Sharpe": [1.0]}, index=["A"])
    portfolio_series = pd.Series(
        [0.1, 0.2], index=pd.Index(["2020-01", "2020-02"]), name="user"
    )
    benchmark_series = pd.Series(
        [0.05], index=pd.Index(["2020-01"], name="month"), name="bench"
    )
    weights_df = pd.DataFrame({"A": [0.6], "B": [0.4]})

    cache_first = {
        "entries": 1,
        "hits": 2,
        "misses": 3,
        "incremental_updates": 4,
    }
    cache_second = {
        "entries": 5.0,
        "hits": 6.0,
        "misses": 7.0,
        "incremental_updates": 8.0,
    }

    # Helper class for tests
    class TruthySeries:
        def __init__(self, series: pd.Series):
            self.series = series

        def __bool__(self) -> bool:
            return True

        def __getattr__(self, name: str):
            return getattr(self.series, name)

    details = {
        "cache": cache_first,
        "nested": [cache_second],
        "portfolio_user_weight": TruthySeries(portfolio_series),
        "benchmarks": {"BMK": benchmark_series},
        "weights_user_weight": weights_df,
    }

    run_result = SimpleNamespace(metrics=metrics_df, details=details, seed=11)

    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "load_market_data_csv",
        lambda path: SimpleNamespace(
            frame=pd.DataFrame({"Date": pd.to_datetime(["2020-01-31"]), "A": [0.0]})
        ),
    )
    monkeypatch.setattr(cli, "run_simulation", lambda cfg, df: run_result)

    summary_calls: list[tuple] = []
    monkeypatch.setattr(
        cli.export,
        "format_summary_text",
        lambda *args: summary_calls.append(args) or "summary text",
    )

    formatter_calls: list[tuple] = []
    monkeypatch.setattr(
        cli.export,
        "make_summary_formatter",
        lambda *args: formatter_calls.append(args) or object(),
    )

    excel_calls: list[tuple] = []
    monkeypatch.setattr(
        cli.export,
        "export_to_excel",
        lambda data, path, default_sheet_formatter: excel_calls.append(
            (data, Path(path), default_sheet_formatter)
        ),
    )

    data_calls: list[tuple] = []
    monkeypatch.setattr(
        cli.export,
        "export_data",
        lambda data, path, formats: data_calls.append(
            (data, Path(path), tuple(formats))
        ),
    )

    bundle_calls: list[tuple] = []
    monkeypatch.setattr(
        "trend_analysis.export.bundle.export_bundle",
        lambda rr, bundle_path: bundle_calls.append((rr, Path(bundle_path))),
    )

    log_calls: list[tuple[object, object]] = []

    def fake_log_step(run_id: str, *args, **kwargs) -> None:
        event_name = kwargs.get("event")
        if event_name is None and args:
            event_name = args[0]
        message = kwargs.get("message")
        if message is None and len(args) > 1:
            message = args[1]
        log_calls.append((event_name, message))

    monkeypatch.setattr(
        "trend_analysis.logging.get_default_log_path",
        lambda run_id: tmp_path / f"{run_id}.jsonl",
    )
    monkeypatch.setattr(
        "trend_analysis.logging.init_run_logger",
        lambda run_id, path: log_calls.append(("init", run_id)),
    )
    monkeypatch.setattr("trend_analysis.logging.log_step", fake_log_step)
    monkeypatch.setattr(cli, "_log_step", fake_log_step)

    monkeypatch.setattr(
        "uuid.uuid4",
        lambda: SimpleNamespace(hex="abcdef1234567890"),
    )

    monkeypatch.setenv("TREND_SEED", "987")

    bundle_path = tmp_path / "bundle-out.zip"
    rc = cli.main(
        [
            "run",
            "-c",
            "cfg.yml",
            "-i",
            "input.csv",
            "--bundle",
            str(bundle_path),
        ]
    )

    out = capsys.readouterr().out

    assert rc == 0
    assert config.seed == 987
    assert "summary text" in out
    assert "Cache statistics" in out
    assert "Bundle written" in out
    assert summary_calls and summary_calls[0][0] is details
    assert isinstance(run_result.portfolio, TruthySeries)
    assert run_result.portfolio.series is portfolio_series
    assert run_result.benchmark is benchmark_series
    assert run_result.weights is weights_df
    assert formatter_calls
    assert excel_calls and excel_calls[0][1] == out_dir / "analysis.xlsx"
    assert data_calls == [(excel_calls[0][0], out_dir / "analysis", ("json",))]
    assert bundle_calls == [(run_result, bundle_path)]
    assert any(event == "cache_stats" for event, _ in log_calls)
