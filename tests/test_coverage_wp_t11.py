"""Coverage tests for work package t11 (trend CLI, report export, LLM proxy)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import trend.cli as cli
import trend.cli_support as cli_support
import trend.commands.report_export as report_export
from trend.cli_helpers import _apply_trend_spec_preset, _apply_universe_mask, _attach_universe_paths
from trend.cli_support import check_environment, extract_cache_stats, find_prior_run
from trend.commands.llm_support import _resolve_llm_provider_config
from trend.commands.report_export import (
    _bundle_config_payload,
    _maybe_write_turnover_csv,
    _persist_turnover_ledger,
    _portfolio_settings,
    _report_pipeline_diagnostic,
    _require_transaction_cost_controls,
    _resolve_export_artifact_paths,
    _write_trend_run_artifacts,
)
from trend.diagnostics import DiagnosticPayload, DiagnosticResult
from trend.mc.viz import TrendCLIError
from trend_analysis.io.market_data import MarketDataValidationError


@pytest.fixture(autouse=True)
def _no_ambient_llm_or_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI sets provider and proxy secrets in the environment (its log shows them as ***), and the
    proxy resolves its upstream key from several of them in a fixed order. Remove every such
    variable before each test; the tests that need one set it themselves."""
    for name in list(os.environ):
        if name.startswith(("TS_", "TREND_", "OPENAI_", "ANTHROPIC_", "LANGSMITH_", "LANGCHAIN_")):
            monkeypatch.delenv(name, raising=False)


def _mde(msg: str = "bad data", issues: list[str] | None = None) -> MarketDataValidationError:
    return MarketDataValidationError(msg, issues=issues or [])


# --- cli_support.check_environment / find_prior_run / extract_cache_stats ---


def test_check_environment_missing_lock(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "nope.lock"
    assert check_environment(missing) == 1
    out = capsys.readouterr().out
    assert f"Lock file not found: {missing}" in out
    assert out.startswith("Python ")


def test_check_environment_match_and_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text("# comment\n\npandas==2.1.0\nnumpy==1.26.0 extra\nbadline\n", encoding="utf-8")
    versions = {"pandas": "2.1.0", "numpy": "9.9.9"}
    monkeypatch.setattr(cli_support.metadata, "version", lambda n: versions[n])
    assert check_environment(lock) == 1
    out = capsys.readouterr().out
    assert "pandas 2.1.0 (expected 2.1.0)" in out
    assert "numpy 9.9.9 (expected 1.26.0)" in out
    assert "Mismatches detected:" in out
    assert "- numpy: installed 9.9.9, expected 1.26.0" in out
    versions["numpy"] = "1.26.0"
    assert check_environment(lock) == 0
    assert "All packages match lockfile." in capsys.readouterr().out


def test_check_environment_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text("ghostpkg==3.2.1\n", encoding="utf-8")

    def _raise(name: str) -> str:
        raise cli_support.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(cli_support.metadata, "version", _raise)
    assert check_environment(lock) == 1
    out = capsys.readouterr().out
    assert "ghostpkg not installed (expected 3.2.1)" in out
    assert "- ghostpkg: installed none, expected 3.2.1" in out


def test_find_prior_run_variants(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[tuple[Path, str]] = []

    def fake_find(out_dir: Path, run_id: str) -> Path:
        seen.append((Path(out_dir), run_id))
        return Path(out_dir) / f"{run_id}.json"

    monkeypatch.setattr(cli_support, "find_existing_run", fake_find)
    cfg_map = SimpleNamespace(export={"directory": "outA", "formats": ["csv"]})
    assert find_prior_run(cfg_map, "r1") == Path("outA/r1.json")
    cfg_obj = SimpleNamespace(export=SimpleNamespace(directory="outB", formats=["json"]))
    assert find_prior_run(cfg_obj, "r2") == Path("outB/r2.json")
    cfg_none = SimpleNamespace(export=None)
    result = find_prior_run(cfg_none, "r3")
    assert result == Path(cli_support.DEFAULT_OUTPUT_DIRECTORY) / "r3.json"
    cfg_defaults = SimpleNamespace(export={"directory": "", "formats": []})
    assert (
        find_prior_run(cfg_defaults, "r4") == Path(cli_support.DEFAULT_OUTPUT_DIRECTORY) / "r4.json"
    )
    cfg_no_formats = SimpleNamespace(export={"directory": "outC", "formats": []})
    assert find_prior_run(cfg_no_formats, "r5") is None
    assert seen == [
        (Path("outA"), "r1"),
        (Path("outB"), "r2"),
        (Path(cli_support.DEFAULT_OUTPUT_DIRECTORY), "r3"),
        (Path(cli_support.DEFAULT_OUTPUT_DIRECTORY), "r4"),
    ]


def test_extract_cache_stats_rejects_nonintegral() -> None:
    payload = {"entries": "x", "hits": 1, "misses": 2, "incremental_updates": 3}
    assert extract_cache_stats(payload) is None


# --- cli_helpers ---


def test_attach_universe_paths_mapping(tmp_path: Path) -> None:
    cfg = SimpleNamespace(data={"keep": 1})
    spec = SimpleNamespace(membership_path=tmp_path / "mem.csv")
    _attach_universe_paths(cfg, spec, csv_path="data.csv")
    assert cfg.data == {
        "keep": 1,
        "universe_membership_path": str(tmp_path / "mem.csv"),
        "csv_path": "data.csv",
    }
    assert cfg.data["keep"] == 1
    cfg2 = SimpleNamespace(data={"csv_path": "orig.csv"})
    _attach_universe_paths(cfg2, spec, csv_path="new.csv")
    assert cfg2.data["csv_path"] == "orig.csv"


def test_attach_universe_paths_none_data() -> None:
    cfg = SimpleNamespace(data=None)
    spec = SimpleNamespace(membership_path="m.csv")
    _attach_universe_paths(cfg, spec, csv_path=None)
    assert cfg.data == {"universe_membership_path": "m.csv"}
    cfg2 = SimpleNamespace(data=None)
    _attach_universe_paths(cfg2, spec, csv_path="c.csv")
    assert cfg2.data == {"universe_membership_path": "m.csv", "csv_path": "c.csv"}
    assert set(cfg2.data) == {"universe_membership_path", "csv_path"}


def test_attach_universe_paths_object_data() -> None:
    class DataSection:
        def __init__(self) -> None:
            self.writes: list[tuple[str, str]] = []

        def __setattr__(self, name: str, value: Any) -> None:
            if name == "writes":
                super().__setattr__(name, value)
            else:
                self.writes.append((name, value))

    data = DataSection()
    cfg = SimpleNamespace(data=data)
    spec = SimpleNamespace(membership_path="m.csv")
    _attach_universe_paths(cfg, spec, csv_path="c.csv")
    assert data.writes == [("universe_membership_path", "m.csv"), ("csv_path", "c.csv")]


def test_attach_universe_paths_setattr_fallbacks() -> None:
    class CfgMappingFallback:
        def __init__(self) -> None:
            object.__setattr__(self, "data", {"keep": 1})

        def __setattr__(self, name: str, value: Any) -> None:
            if name == "data":
                raise TypeError("blocked")
            object.__setattr__(self, name, value)

    spec = SimpleNamespace(membership_path="mem.csv")
    mapping_cfg = CfgMappingFallback()
    _attach_universe_paths(mapping_cfg, spec, csv_path="data.csv")
    assert mapping_cfg.data == {
        "keep": 1,
        "universe_membership_path": "mem.csv",
        "csv_path": "data.csv",
    }

    class CfgNoneFallback:
        def __setattr__(self, name: str, value: Any) -> None:
            if name == "data":
                raise TypeError("blocked")
            object.__setattr__(self, name, value)

    none_cfg = CfgNoneFallback()
    _attach_universe_paths(none_cfg, spec, csv_path=None)
    assert none_cfg.data == {"universe_membership_path": "mem.csv"}

    class SlottedMembership:
        __slots__ = ("universe_membership_path",)

    csv_cfg = SimpleNamespace(data=SlottedMembership())
    _attach_universe_paths(csv_cfg, spec, csv_path="c.csv")
    assert csv_cfg.data.universe_membership_path == "mem.csv"
    assert not hasattr(csv_cfg.data, "csv_path")


def test_attach_universe_paths_frozen_fallback() -> None:
    class SlottedData:
        __slots__ = ("x",)

        def __init__(self) -> None:
            self.x = 1

    cfg = SimpleNamespace(data=SlottedData())
    spec = SimpleNamespace(membership_path="m.csv")
    _attach_universe_paths(cfg, spec, csv_path="c.csv")
    assert getattr(cfg.data, "x", None) == 1
    assert not hasattr(cfg.data, "universe_membership_path")
    assert not hasattr(cfg.data, "csv_path")


def test_apply_trend_spec_preset_nonmapping_signals() -> None:
    class Preset:
        name = "fast"

        def as_signal_config(self) -> dict[str, Any]:
            return {"window": 5}

    cfg = SimpleNamespace(signals="nope")
    _apply_trend_spec_preset(cfg, Preset())
    assert cfg.signals == {"window": 5}
    assert cfg.trend_spec_preset == "fast"


def test_apply_universe_mask_empty_returns_same() -> None:
    df = pd.DataFrame({"Date": [1], "A": [2.0]})
    empty_mask = pd.DataFrame()
    assert empty_mask.empty
    out = _apply_universe_mask(df, empty_mask, date_column="Date")
    assert out is df and out.loc[0, "A"] == 2.0


# --- llm_support ---


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "TREND_LLM_PROVIDER",
        "TREND_LLM_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "TREND_LLM_MODEL",
        "TREND_LLM_BASE_URL",
        "TREND_LLM_ORG",
        "TREND_LLM_MAX_RETRIES",
        "TREND_LLM_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_resolve_llm_defaults_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-open")
    cfg = _resolve_llm_provider_config()
    assert (cfg.provider, cfg.api_key, cfg.model) == ("openai", "sk-open", "gpt-4o-mini")


def test_resolve_llm_anthropic_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("TREND_LLM_MODEL", "claude-x")
    monkeypatch.setenv("TREND_LLM_BASE_URL", "http://proxy:9")
    monkeypatch.setenv("TREND_LLM_ORG", "org1")
    monkeypatch.setenv("TREND_LLM_MAX_RETRIES", "4")
    monkeypatch.setenv("TREND_LLM_TIMEOUT", "12.5")
    cfg = _resolve_llm_provider_config("ANTHROPIC")
    assert cfg.provider == "anthropic"
    assert cfg.api_key == "sk-ant"
    assert cfg.model == "claude-x"
    assert cfg.base_url == "http://proxy:9"
    assert cfg.organization == "org1"
    assert cfg.max_retries == 4
    assert cfg.timeout == 12.5


def test_resolve_llm_unknown_and_bad_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    with pytest.raises(TrendCLIError, match="Unknown LLM provider 'bogus'"):
        _resolve_llm_provider_config("bogus")
    monkeypatch.setenv("TREND_LLM_MAX_RETRIES", "abc")
    with pytest.raises(TrendCLIError, match="TREND_LLM_MAX_RETRIES must be an integer"):
        _resolve_llm_provider_config("ollama")
    monkeypatch.setenv("TREND_LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("TREND_LLM_TIMEOUT", "xyz")
    with pytest.raises(TrendCLIError, match="TREND_LLM_TIMEOUT must be a number"):
        _resolve_llm_provider_config("ollama")
    monkeypatch.delenv("TREND_LLM_TIMEOUT")
    monkeypatch.setenv("TREND_LLM_API_KEY", "k1")
    cfg = _resolve_llm_provider_config("ollama", model="m9")
    assert cfg.model == "m9"
    assert cfg.api_key == "k1"
    assert cfg.max_retries == 2


# --- report_export ---


def test_report_pipeline_diagnostic(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(report_export, "maybe_log_step", lambda *a, **k: calls.append((a, k)))
    diag = DiagnosticPayload(
        reason_code="SKIP1", message="no data", context={"a": 1, 2: "bad"}  # type: ignore[dict-item]
    )
    _report_pipeline_diagnostic(diag, structured_log=True, run_id="run9")
    err = capsys.readouterr()
    assert err.out == "Pipeline skipped (SKIP1): no data\n" and err.err == ""
    assert len(calls) == 1
    assert calls[0][0] == (True, "run9", "pipeline_diagnostic", "no data")
    assert calls[0][1] == {"reason_code": "SKIP1", "a": 1}


def test_resolve_export_artifact_paths_dedup() -> None:
    out = Path("/tmp/o")
    paths = _resolve_export_artifact_paths(
        out, "f", ["metrics", "metrics"], ["", "CSV", "csv", "xlsx", "XLSX"]
    )
    assert paths == [out / "f_metrics.csv", out / "f.xlsx"]


def test_write_trend_artifacts_no_export(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SimpleNamespace(export={"directory": "out", "formats": []})
    result = SimpleNamespace(metrics=pd.DataFrame(), details={})
    monkeypatch.setattr(
        report_export,
        "write_run_artifacts",
        lambda **k: (_ for _ in ()).throw(AssertionError("should not write")),
    )
    assert (
        _write_trend_run_artifacts(
            cfg=cfg,
            result=result,  # type: ignore[arg-type]
            config_path=Path("c.yml"),
            input_path=Path("i.csv"),
            data_frame=pd.DataFrame(),
            run_id="r",
            structured_log=False,
        )
        is None
    )


def test_write_trend_artifacts_plain_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(report_export, "load_config_yaml", lambda p: {"version": "1"})
    monkeypatch.setattr(report_export.export, "append_narrative_section", lambda *a, **k: None)
    monkeypatch.setattr(report_export, "_summary_text", lambda cfg, details: "SUM")

    def fake_write(**kwargs: Any) -> Path:
        captured.update(kwargs)
        return tmp_path / "manifest"

    monkeypatch.setattr(report_export, "write_run_artifacts", fake_write)
    monkeypatch.setattr(report_export, "write_run_envelope", lambda *a, **k: tmp_path / "env.json")
    monkeypatch.setattr(report_export, "maybe_log_step", lambda *a, **k: None)
    cfg = object()  # no model_dump, no __dict__
    result = SimpleNamespace(metrics=pd.DataFrame(), details={"x": 1})
    out = _write_trend_run_artifacts(
        cfg=cfg,
        result=result,  # type: ignore[arg-type]
        config_path=tmp_path / "c.yml",
        input_path=tmp_path / "i.csv",
        data_frame=pd.DataFrame(),
        run_id="r1",
        structured_log=False,
    )
    assert out == tmp_path / "manifest"
    assert captured["config"] is cfg
    assert captured["run_id"] == "r1"
    assert captured["config_path"] == str(tmp_path / "c.yml")


def test_bundle_config_payload_variants() -> None:
    class WithDump:
        def model_dump(self) -> dict[str, Any]:
            return {"a": 1, "trend_spec": 2, "_trend_run_spec": 3}

    dumped = _bundle_config_payload(WithDump())
    assert dumped == {"a": 1} and "trend_spec" not in dumped and "_trend_run_spec" not in dumped
    ns = _bundle_config_payload(SimpleNamespace(b=2))
    assert ns == {"b": 2} and list(ns) == ["b"]
    mapping = _bundle_config_payload({"c": 3, "backtest_spec": 9})
    assert mapping == {"c": 3} and "backtest_spec" not in mapping
    assert _bundle_config_payload(42) == {}


def test_maybe_write_turnover_csv_branches(tmp_path: Path) -> None:
    r = _maybe_write_turnover_csv(tmp_path, "nope")
    assert r.value is None and r.diagnostic is not None
    assert r.diagnostic.context == {"details_type": "str"}
    r = _maybe_write_turnover_csv(
        tmp_path, {"risk_diagnostics": {"turnover": {"2021-01-01": 0.5, "2021-01-02": 0.25}}}
    )
    assert r.value == tmp_path / "turnover.csv"
    assert r.value.read_text(encoding="utf-8").splitlines()[0] == "Date,turnover"
    r = _maybe_write_turnover_csv(tmp_path, {"risk_diagnostics": {"turnover": [0.1, 0.2]}})
    assert r.value == tmp_path / "turnover.csv"
    r = _maybe_write_turnover_csv(tmp_path, {"risk_diagnostics": {"turnover": ["a"]}})
    assert r.value is None and r.diagnostic.context == {"turnover_type": "list"}
    r = _maybe_write_turnover_csv(
        tmp_path, {"risk_diagnostics": {"turnover": pd.Series([], dtype=float)}}
    )
    assert r.value is None and r.diagnostic.context == {"turnover_type": "Series"}
    r = _maybe_write_turnover_csv(tmp_path, {"risk_diagnostics": {"turnover": 5}})
    assert r.value is None and r.diagnostic.context == {"turnover_type": "int"}


def test_portfolio_settings_and_cost_controls() -> None:
    assert _portfolio_settings(SimpleNamespace(portfolio={"a": 1})) == {"a": 1}
    assert _portfolio_settings(SimpleNamespace(portfolio=SimpleNamespace(b=2))) == {"b": 2}
    assert _portfolio_settings(SimpleNamespace(portfolio=42)) == {}
    with pytest.raises(
        TrendCLIError, match=r"Configuration must define portfolio\.cost_model\.per_trade_bps\."
    ):
        _require_transaction_cost_controls(SimpleNamespace(portfolio={"cost_model": {}}))
    with pytest.raises(
        TrendCLIError, match=r"portfolio\.cost_model\.per_trade_bps must be numeric"
    ):
        _require_transaction_cost_controls(
            SimpleNamespace(portfolio={"cost_model": {"per_trade_bps": "x", "half_spread_bps": 1}})
        )
    with pytest.raises(
        TrendCLIError, match=r"portfolio\.cost_model\.per_trade_bps must be a finite number"
    ):
        _require_transaction_cost_controls(
            SimpleNamespace(
                portfolio={"cost_model": {"per_trade_bps": float("inf"), "half_spread_bps": 1}}
            )
        )
    with pytest.raises(
        TrendCLIError, match=r"portfolio\.cost_model\.half_spread_bps cannot be negative"
    ):
        _require_transaction_cost_controls(
            SimpleNamespace(portfolio={"cost_model": {"per_trade_bps": 1, "half_spread_bps": -1}})
        )
    _require_transaction_cost_controls(
        SimpleNamespace(portfolio={"cost_model": {"per_trade_bps": 0, "half_spread_bps": "2.5"}})
    )


def test_persist_turnover_ledger_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    r = _persist_turnover_ledger("run1", {"risk_diagnostics": {"turnover": None}})
    assert r.diagnostic is not None and r.diagnostic.context == {"turnover_type": None}
    r = _persist_turnover_ledger(
        "run1", {"risk_diagnostics": {"turnover": pd.Series([], dtype=float)}}
    )
    assert r.diagnostic.context == {"turnover_type": "Series"}
    r = _persist_turnover_ledger("run1", {"risk_diagnostics": {"turnover": {}}})
    assert r.diagnostic.context == {"turnover_type": "Mapping"}
    r = _persist_turnover_ledger("run1", {"risk_diagnostics": {"turnover": []}})
    assert r.diagnostic.context == {"turnover_type": "Sequence"}
    r = _persist_turnover_ledger("run1", {"risk_diagnostics": {"turnover": 3.5}})
    assert r.diagnostic.context == {"turnover_type": "float"}
    details = {
        "risk_diagnostics": {"turnover": pd.Series([0.3], index=pd.to_datetime(["2021-01-31"]))}
    }
    r = _persist_turnover_ledger("runZ", details)
    expected = Path("perf") / "runZ" / "turnover.csv"
    assert r.value == expected
    assert (tmp_path / expected).exists()
    assert f"Turnover ledger written to {expected}" in capsys.readouterr().out


def test_persist_turnover_ledger_propagates_write_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    details = {
        "risk_diagnostics": {
            "turnover": pd.Series([0.2], index=pd.to_datetime(["2021-01-31"])),
        }
    }
    write_diag = DiagnosticPayload(reason_code="X", message="write failed", context={})
    monkeypatch.setattr(
        report_export,
        "_maybe_write_turnover_csv",
        lambda directory, det: DiagnosticResult(value=None, diagnostic=write_diag),
    )
    r = _persist_turnover_ledger("runW", details)
    assert r.value is None and r.diagnostic is write_diag


def test_run_pipeline_diagnostic_early_return(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import trend_analysis.api as api

    monkeypatch.setattr(report_export, "_require_transaction_cost_controls", lambda cfg: None)
    monkeypatch.setattr(
        report_export,
        "_init_perf_logger",
        lambda: report_export.DiagnosticResult(
            value=None,
            diagnostic=DiagnosticPayload(reason_code="X", message="perf off"),
        ),
    )
    monkeypatch.setattr(report_export, "working_run_id", lambda cfg, src: "rid42")
    diag = DiagnosticPayload(reason_code="EMPTY", message="empty result")
    result = SimpleNamespace(metrics=pd.DataFrame(), details={}, diagnostic=diag)
    monkeypatch.setattr(report_export, "run_simulation", lambda cfg, df: result)
    monkeypatch.setattr(report_export, "maybe_log_step", lambda *a, **k: None)
    cfg = SimpleNamespace(portfolio={})
    out_result, run_id, log_path = report_export._run_pipeline(
        cfg,
        pd.DataFrame({"A": [1]}),
        source_path=None,
        log_file=None,
        structured_log=False,
        bundle=None,
    )
    assert out_result is result and run_id == "rid42" and log_path is None
    assert api is not None


# --- cli.py helpers ---


def test_run_environment_check_delegates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text("# empty lock\n", encoding="utf-8")
    monkeypatch.setattr(cli_support, "LOCK_PATH", lock)
    assert cli._run_environment_check() == 0
    assert "All packages match lockfile." in capsys.readouterr().out


def test_resolve_returns_duplicate_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "c.yml"
    cfg_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli, "proj_path", lambda: cfg_dir)

    class Cfg:
        data = {"csv_path": "missing.csv"}

    resolved = cli._resolve_returns_path(cfg_path, Cfg(), None)
    assert resolved == (cfg_dir / "missing.csv").resolve()
    assert resolved.name == "missing.csv"


def test_ensure_dataframe_object_config_and_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_obj = SimpleNamespace(missing_policy=None, missing_limit=None, date_column="When")
    cfg = SimpleNamespace(data=data_obj)
    monkeypatch.setattr(cli, "_confirm_ui_date_fixes", lambda *a, **k: None)

    def bad_load(*a: Any, **k: Any) -> Any:
        raise _mde("broken file", ["row 2 bad"])

    monkeypatch.setattr(cli, "load_ui_dataset", bad_load)
    with pytest.raises(TrendCLIError, match="broken file"):
        cli._ensure_dataframe(Path("x.csv"), config=cfg, auto_fix_dates=True, yes=True)
    frame = pd.DataFrame({"When": ["2021-01-01"], "A": [1.0]}).set_index("When")
    summary = SimpleNamespace(corrected_dates=0, dropped_rows=2, dropped_columns=["Date2"])
    monkeypatch.setattr(cli, "load_ui_dataset", lambda *a, **k: (frame, None, summary))
    out = cli._ensure_dataframe(Path("x.csv"), config=cfg, auto_fix_dates=True, yes=True)
    assert (
        capsys.readouterr().out
        == "Applied UI-style date fixes: 2 row(s) dropped, dropped date-named column(s): Date2\n"
    )
    assert list(out.columns) == ["Date", "A"]
    assert out.loc[0, "A"] == 1.0


def test_load_ui_payload_branches(tmp_path: Path) -> None:
    missing = tmp_path / "no.json"
    with pytest.raises(TrendCLIError, match="Unable to load Streamlit JSON export:"):
        cli._load_ui_payload(missing)
    bad = tmp_path / "bad.json"
    bad.write_text("[1,2]", encoding="utf-8")
    with pytest.raises(
        TrendCLIError, match="Streamlit JSON export must contain an object at the root"
    ):
        cli._load_ui_payload(bad)
    nested = tmp_path / "n.json"
    nested.write_text(json.dumps({"model_state": {"a": 1}, "z": 2}), encoding="utf-8")
    payload, state = cli._load_ui_payload(nested)
    assert payload == {"model_state": {"a": 1}, "z": 2} and state == {"a": 1}
    payload2, state2 = cli._load_ui_payload(nested, {"flat": True})
    assert payload2 == {"model_state": {"flat": True}} and state2 == {"flat": True}


def test_read_ui_config_payload_bad(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text("{not json", encoding="utf-8")
    assert cli._read_ui_config_payload(p) is None
    p.write_text("[1]", encoding="utf-8")
    assert cli._read_ui_config_payload(p) is None
    p.write_text(json.dumps({"version": "1", "data": {}}), encoding="utf-8")
    assert cli._read_ui_config_payload(p) is None


def _issues(has: bool = False, rows: int = 0, corr: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        has_corrections=has, total_droppable_rows=rows, corrections=[object()] * corr
    )


def test_confirm_ui_date_fixes_all(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli, "inspect_ui_date_issues", lambda *a, **k: (_ for _ in ()).throw(_mde("nope", ["i1"]))
    )
    with pytest.raises(TrendCLIError, match="nope"):
        cli._confirm_ui_date_fixes(Path("d.csv"), yes=False)
    monkeypatch.setattr(cli, "inspect_ui_date_issues", lambda *a, **k: _issues())
    cli._confirm_ui_date_fixes(Path("d.csv"), yes=False)
    monkeypatch.setattr(
        cli, "inspect_ui_date_issues", lambda *a, **k: _issues(has=True, rows=1, corr=1)
    )
    cli._confirm_ui_date_fixes(Path("d.csv"), yes=True)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    with pytest.raises(
        TrendCLIError,
        match=r"Date corrections require confirmation\. Re-run with --yes to approve\.",
    ):
        cli._confirm_ui_date_fixes(Path("d.csv"), yes=False)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    with pytest.raises(TrendCLIError, match="Date corrections cancelled"):
        cli._confirm_ui_date_fixes(Path("d.csv"), yes=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": " YES ")
    cli._confirm_ui_date_fixes(Path("d.csv"), yes=False)


def test_prepare_ui_inputs_errors_and_print(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(
        config=str(tmp_path / "u.json"), returns=None, auto_fix_dates=False, yes=False, seed=None
    )
    with pytest.raises(
        TrendCLIError, match=r"Streamlit JSON replay requires --input/--returns data"
    ):
        cli._prepare_ui_command_inputs(args, prepare_config=lambda c: None)
    data_path = tmp_path / "d.csv"
    data_path.write_text("x", encoding="utf-8")
    args.returns = str(data_path)
    args.auto_fix_dates = True
    confirmed: list[Any] = []
    monkeypatch.setattr(cli, "_confirm_ui_date_fixes", lambda p, yes: confirmed.append((p, yes)))

    def bad_load(*a: Any, **k: Any) -> Any:
        raise _mde("ui broken", ["e1"])

    monkeypatch.setattr(cli, "load_ui_dataset", bad_load)
    payload = {"model_state": {"info_ratio_benchmark": " Bench "}}
    with pytest.raises(TrendCLIError, match="ui broken"):
        cli._prepare_ui_command_inputs(args, prepare_config=lambda c: None, ui_payload=payload)
    assert confirmed == [(data_path.resolve(), False)]
    returns = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2021-01-01"]))
    meta = SimpleNamespace(frequency="M")
    summary = SimpleNamespace(corrected_dates=1, dropped_rows=1, dropped_columns=["Dx"])
    monkeypatch.setattr(cli, "load_ui_dataset", lambda *a, **k: (returns, meta, summary))
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        cli, "build_config_from_ui_state", lambda **k: captured.update(k) or SimpleNamespace(seed=3)
    )
    monkeypatch.setattr(cli, "ensure_run_spec", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_determine_seed", lambda cfg, s: 9)
    prepared_calls: list[Any] = []
    out = cli._prepare_ui_command_inputs(
        args, prepare_config=lambda c: prepared_calls.append(c), ui_payload=payload
    )
    assert capsys.readouterr().out == (
        "Applied UI-style date fixes: 1 date correction(s), 1 row(s) dropped, "
        "dropped date-named column(s): Dx\n"
    )
    assert captured["benchmark"] == "Bench"
    assert out.seed == 9
    assert prepared_calls[0].seed == 3


def test_load_configuration_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = tmp_path / "c.yml"
    p.write_text("x: 1", encoding="utf-8")
    monkeypatch.setattr(
        cli, "load_schema_config", lambda path: (_ for _ in ()).throw(ValueError("schema boom"))
    )
    with pytest.raises(TrendCLIError, match="schema boom"):
        cli._load_configuration(str(p))
    monkeypatch.setattr(cli, "load_schema_config", lambda path: {"x": 1})
    monkeypatch.setattr(
        cli,
        "load_core_config",
        lambda path: (_ for _ in ()).throw(cli.CoreConfigError("core boom")),
    )
    with pytest.raises(TrendCLIError, match="core boom"):
        cli._load_configuration(str(p))
    monkeypatch.setattr(cli, "load_core_config", lambda path: None)
    monkeypatch.setattr(
        cli,
        "validate_config",
        lambda *a, **k: SimpleNamespace(valid=False),
    )
    monkeypatch.setattr(cli, "format_validation_messages", lambda v: ["field x bad"])
    with pytest.raises(TrendCLIError, match=r"Config validation failed:\nfield x bad"):
        cli._load_configuration(str(p))


# --- main() branches ---


def test_main_extra_args_and_unknown(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_maybe_handle_nl_replay", lambda argv: None)
    assert cli.main(["run", "--config", "x.yml", "stray"]) == 2
    assert "Error: Unexpected arguments: stray" in capsys.readouterr().err


def test_main_check_and_app_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_maybe_handle_nl_replay", lambda argv: None)
    events: list[str] = []
    trackers: list[Any] = []
    monkeypatch.setattr(cli, "run_check_command", lambda environment_check: environment_check())
    monkeypatch.setattr(cli, "_run_environment_check", lambda: 5)
    assert cli.main(["check"]) == 5
    monkeypatch.setattr(
        cli,
        "activate_config_coverage",
        lambda t: (events.append("act"), trackers.append(t)),
    )
    monkeypatch.setattr(cli, "deactivate_config_coverage", lambda: events.append("deact"))

    class Tracker:
        def format_report(self) -> str:
            return "COVREPORT"

    monkeypatch.setattr(cli, "ConfigCoverageTracker", Tracker)

    class CheckCoverageParser:
        def parse_known_args(self, argv: list[str]) -> tuple[Any, list[str]]:
            return SimpleNamespace(subcommand="check", config_coverage=True), []

    monkeypatch.setattr(cli, "build_parser", lambda prog="trend": CheckCoverageParser())
    assert cli.main(["check"]) == 5
    assert events == ["act", "deact"]
    assert trackers[-1].format_report() == "COVREPORT"
    events.clear()
    trackers.clear()
    monkeypatch.setattr(cli, "run_app_command", lambda *a, **k: 3)

    class AppCoverageParser:
        def parse_known_args(self, argv: list[str]) -> tuple[Any, list[str]]:
            return SimpleNamespace(subcommand="app", config_coverage=True), []

    monkeypatch.setattr(cli, "build_parser", lambda prog="trend": AppCoverageParser())
    assert cli.main(["app"]) == 3
    assert events == ["act", "deact"]
    assert trackers[-1].format_report() == "COVREPORT"


def test_main_quick_report_and_explain_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_maybe_handle_nl_replay", lambda argv: None)
    monkeypatch.setattr(cli, "activate_config_coverage", lambda t: None)
    monkeypatch.setattr(cli, "deactivate_config_coverage", lambda: None)

    class Tracker:
        def format_report(self) -> str:
            return "R"

    monkeypatch.setattr(cli, "ConfigCoverageTracker", Tracker)
    got: list[list[str]] = []
    monkeypatch.setattr(cli, "quick_summary_main", lambda a: got.append(a) or 0)
    out_file = tmp_path / "o.html"
    real_build_parser = cli.build_parser
    deactivate_events: list[str] = []
    monkeypatch.setattr(
        cli, "deactivate_config_coverage", lambda: deactivate_events.append("deact")
    )

    class QuickCoverageParser:
        def parse_known_args(self, argv: list[str]) -> tuple[Any, list[str]]:
            return (
                SimpleNamespace(
                    subcommand="quick-report",
                    config_coverage=True,
                    run_id=None,
                    artifacts=None,
                    base_dir=None,
                    config=None,
                    output=out_file,
                ),
                [],
            )

    monkeypatch.setattr(cli, "build_parser", lambda prog="trend": QuickCoverageParser())
    assert cli.main(["quick-report", "--output", str(out_file)]) == 0
    assert deactivate_events == ["deact"]
    assert got == [["--output", str(out_file)]]
    monkeypatch.setattr(cli, "build_parser", real_build_parser)
    # explain with no metrics, with and without output
    monkeypatch.setattr(cli, "_resolve_explain_details_path", lambda a: tmp_path / "d.json")
    monkeypatch.setattr(cli, "_load_explain_details", lambda p: {})
    monkeypatch.setattr(cli, "_resolve_explain_questions", lambda a: [])
    monkeypatch.setattr(cli, "_infer_explain_run_id", lambda *a: "runE")
    monkeypatch.setattr(cli, "extract_metric_catalog", lambda d: [])
    monkeypatch.setattr(cli, "compact_metric_catalog", lambda e, questions: [])
    monkeypatch.setattr(cli, "format_metric_catalog", lambda e: "")
    monkeypatch.setattr(cli, "_finalize_explanation_text", lambda t, i: t)
    assert cli.main(["explain"]) == 0
    assert "No metrics were detected in the analysis output." in capsys.readouterr().out
    written: list[Any] = []
    monkeypatch.setattr(cli, "_build_explain_artifact_payload", lambda **k: k)
    monkeypatch.setattr(cli, "_write_explain_artifacts", lambda **k: written.append(k))
    assert cli.main(["explain", "--output", str(tmp_path / "e.txt")]) == 0
    assert written[0]["run_id"] == "runE"
    assert written[0]["text"] == "No metrics were detected in the analysis output."


def _nl_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(cli, "_maybe_handle_nl_replay", lambda argv: None)
    cfg_file = tmp_path / "in.yml"
    cfg_file.write_text("a: 1\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_load_nl_config", lambda p: {"a": 1})
    return cfg_file


def test_main_nl_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_file = _nl_setup(monkeypatch, tmp_path)
    assert cli.main(["nl", "do x", "--in", str(tmp_path / "missing.yml")]) == 2
    assert "Input config not found:" in capsys.readouterr().err
    monkeypatch.setattr(
        cli, "_apply_nl_instruction", lambda *a, **k: ({"p": 1}, {"a": 2}, "", "m", 0.1)
    )
    assert cli.main(["nl", "do x", "--in", str(cfg_file), "--run", "--diff"]) == 2
    assert "--run cannot be combined with --diff or --dry-run" in capsys.readouterr().err
    assert cli.main(["nl", "do x", "--in", str(cfg_file), "--diff"]) == 0
    assert capsys.readouterr().out == "No changes.\n"
    import yaml as yaml_mod

    assert cli.main(["nl", "do x", "--in", str(cfg_file), "--dry-run"]) == 0
    assert capsys.readouterr().out == yaml_mod.safe_dump(
        {"a": 2}, sort_keys=False, default_flow_style=False
    )


def test_main_nl_run_validation_and_load_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_file = _nl_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_validate_nl_run_config", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_confirm_risky_patch", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_log_nl_operation", lambda **k: None)
    monkeypatch.setattr(
        cli, "_apply_nl_instruction", lambda *a, **k: ({"p": 1}, {"a": 2}, "d", "m", 0.1)
    )
    monkeypatch.setattr(
        cli, "validate_config", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("val boom"))
    )
    assert cli.main(["nl", "x", "--in", str(cfg_file), "--run"]) == 2
    assert "Error: val boom" in capsys.readouterr().err
    monkeypatch.setattr(cli, "validate_config", lambda *a, **k: SimpleNamespace(valid=False))
    monkeypatch.setattr(cli, "format_validation_messages", lambda v: ["bad field"])
    assert cli.main(["nl", "x", "--in", str(cfg_file), "--run"]) == 2
    assert "Config validation failed:\nbad field" in capsys.readouterr().err
    monkeypatch.setattr(cli, "validate_config", lambda *a, **k: SimpleNamespace(valid=True))
    monkeypatch.setattr(
        cli, "load_config", lambda p: (_ for _ in ()).throw(ValueError("load boom"))
    )
    assert cli.main(["nl", "x", "--in", str(cfg_file), "--run"]) == 2
    assert "Error: load boom" in capsys.readouterr().err


def test_main_nl_run_pipeline_error_and_success_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_file = _nl_setup(monkeypatch, tmp_path)
    out_file = tmp_path / "out.yml"
    monkeypatch.setattr(cli, "_validate_nl_run_config", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_confirm_risky_patch", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_log_nl_operation", lambda **k: None)
    monkeypatch.setattr(
        cli, "_apply_nl_instruction", lambda *a, **k: ({"p": 1}, {"a": 2}, "d", "m", 0.1)
    )
    monkeypatch.setattr(cli, "validate_config", lambda *a, **k: SimpleNamespace(valid=True))
    monkeypatch.setattr(cli, "load_config", lambda p: SimpleNamespace(seed=1))
    monkeypatch.setattr(cli, "ensure_run_spec", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_resolve_returns_path", lambda *a: tmp_path / "r.csv")
    monkeypatch.setattr(cli, "_ensure_dataframe", lambda p: pd.DataFrame())
    monkeypatch.setattr(cli, "_determine_seed", lambda c, s: 1)
    monkeypatch.setattr(cli, "_print_summary", lambda *a: None)
    monkeypatch.setattr(
        cli, "_run_pipeline", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("pipe boom"))
    )
    assert cli.main(["nl", "x", "--in", str(cfg_file), "--out", str(out_file), "--run"]) == 2
    assert "Error: pipe boom" in capsys.readouterr().err
    result = SimpleNamespace()
    monkeypatch.setattr(cli, "_run_pipeline", lambda *a, **k: (result, "rid", Path("log.json")))
    assert cli.main(["nl", "x", "--in", str(cfg_file), "--out", str(out_file), "--run"]) == 0
    out = capsys.readouterr().out
    assert f"Updated config written: {out_file}" in out
    assert "Structured log: log.json" in out


def test_main_coverage_finalize_via_mc_and_wrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_maybe_handle_nl_replay", lambda argv: None)
    events: list[str] = []

    class Tracker:
        def format_report(self) -> str:
            return "COVREPORT"

    monkeypatch.setattr(cli, "ConfigCoverageTracker", Tracker)
    monkeypatch.setattr(cli, "activate_config_coverage", lambda t: events.append("act"))
    monkeypatch.setattr(cli, "deactivate_config_coverage", lambda: events.append("deact"))
    monkeypatch.setattr(cli, "handle_mc_command", lambda a: 4)
    real_build_parser = cli.build_parser

    class McCoverageParser:
        def parse_known_args(self, argv: list[str]) -> tuple[Any, list[str]]:
            return SimpleNamespace(subcommand="mc", config_coverage=True, mc_command="viz"), []

    monkeypatch.setattr(cli, "build_parser", lambda prog="trend": McCoverageParser())
    assert cli.main(["mc", "viz"]) == 4
    assert "COVREPORT" in capsys.readouterr().out
    assert events == ["act", "deact"]
    # wrap_config_for_coverage via run path prepare_config
    monkeypatch.setattr(cli, "build_parser", real_build_parser)
    wrapped: list[Any] = []
    monkeypatch.setattr(cli, "wrap_config_for_coverage", lambda c, t: wrapped.append(c))

    def fake_prepare(args: Any, **kwargs: Any) -> Any:
        kwargs["prepare_config"](SimpleNamespace(tag="cfgX"))
        return SimpleNamespace(
            cfg_path=tmp_path / "c.yml",
            cfg=SimpleNamespace(),
            returns_path=tmp_path / "r.csv",
            returns_df=pd.DataFrame(),
            seed=1,
        )

    monkeypatch.setattr(cli, "prepare_command_inputs", fake_prepare)
    monkeypatch.setattr(cli, "run_analysis_command", lambda *a, **k: 0)
    cfg_file = tmp_path / "c.yml"
    cfg_file.write_text("a: 1", encoding="utf-8")
    assert cli.main(["run", "--config", str(cfg_file), "--config-coverage"]) == 0
    assert wrapped[0].tag == "cfgX"
    capsys.readouterr()


def test_main_unknown_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_maybe_handle_nl_replay", lambda argv: None)
    real_parser = cli.build_parser()
    deactivated: list[str] = []
    monkeypatch.setattr(cli, "activate_config_coverage", lambda t: None)
    monkeypatch.setattr(cli, "deactivate_config_coverage", lambda: deactivated.append("deact"))

    class FakeParser:
        def parse_known_args(self, argv: list[str]) -> tuple[Any, list[str]]:
            return SimpleNamespace(subcommand="bogus", config_coverage=True), []

    monkeypatch.setattr(cli, "build_parser", lambda prog="trend": FakeParser())
    assert cli.main(["bogus"]) == 2
    assert "Error: Unknown command: bogus" in capsys.readouterr().err
    assert deactivated == ["deact"]
    assert real_parser is not None


def test_main_file_not_found_deactivates_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_maybe_handle_nl_replay", lambda argv: None)
    deactivated: list[str] = []
    activated: list[str] = []
    monkeypatch.setattr(cli, "activate_config_coverage", lambda t: activated.append("act"))
    monkeypatch.setattr(cli, "deactivate_config_coverage", lambda: deactivated.append("deact"))
    cfg_file = tmp_path / "c.yml"
    cfg_file.write_text("a: 1", encoding="utf-8")

    def raise_fnf(*a: Any, **k: Any) -> Any:
        raise FileNotFoundError("artifact missing")

    monkeypatch.setattr(cli, "prepare_command_inputs", raise_fnf)
    assert cli.main(["run", "--config", str(cfg_file), "--config-coverage"]) == 2
    err = capsys.readouterr().err
    assert err == "Error: artifact missing\n"
    assert activated == ["act"] and deactivated == ["deact"]


# --- llm_proxy server (in-process, stubbed deps; no sockets) ---


def _fake_proxy(monkeypatch: pytest.MonkeyPatch, modules: dict[str, Any]) -> Any:
    import trend_analysis.llm_proxy.server as srv

    for name, val in modules.items():
        monkeypatch.setattr(srv, name, val, raising=False)
    monkeypatch.setattr(srv, "_DEPS_AVAILABLE", True, raising=False)
    return srv


class _FakeHTTPError(Exception):
    pass


class _FakeTimeout(_FakeHTTPError):
    pass


class _FakeReqErr(_FakeHTTPError):
    pass


def _proxy_fakes() -> tuple[dict[str, Any], dict[str, Any]]:
    state: dict[str, Any] = {"routes": {}}

    class FakeApp:
        def __init__(self, title: str, version: str) -> None:
            state["title"] = title

        def get(self, path: str) -> Any:
            def deco(fn: Any) -> Any:
                state["routes"][("GET", path)] = fn
                return fn

            return deco

        def api_route(self, path: str, methods: list[str]) -> Any:
            def deco(fn: Any) -> Any:
                state["routes"][("FWD", path)] = fn
                return fn

            return deco

    class FakeHTTPException(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FakeResponse:
        def __init__(self, content: Any, status_code: int, headers: dict[str, str]) -> None:
            self.content = content
            self.status_code = status_code
            self.headers = headers

    class FakeAsyncClient:
        def __init__(self, timeout: int) -> None:
            state["timeout"] = timeout

        async def request(
            self, method: str, url: str, headers: dict[str, str], content: bytes
        ) -> Any:
            state["req"] = (method, url, headers, content)
            return SimpleNamespace(
                content=b"OK", status_code=201, headers={"Content-Length": "2", "X-Keep": "yes"}
            )

        async def aclose(self) -> None:
            state["closed"] = True

    httpx_mod = SimpleNamespace(
        AsyncClient=FakeAsyncClient,
        TimeoutException=_FakeTimeout,
        RequestError=_FakeReqErr,
        HTTPError=_FakeHTTPError,
    )
    mods = {
        "FastAPI": FakeApp,
        "HTTPException": FakeHTTPException,
        "Request": object,
        "Response": FakeResponse,
        "httpx": httpx_mod,
        "uvicorn": SimpleNamespace(),
    }
    return mods, state


def test_proxy_init_routes_health_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    mods, state = _proxy_fakes()
    srv = _fake_proxy(monkeypatch, mods)
    monkeypatch.delenv("TS_LLM_PROXY_TOKEN", raising=False)
    monkeypatch.delenv("TS_LLM_PROXY_UPSTREAM", raising=False)
    proxy = srv.LLMProxy("https://up.example/")
    assert proxy.upstream_base == "https://up.example"
    assert state["title"] == "LLM Proxy"
    assert asyncio.run(state["routes"][("GET", "/health")]()) == {"status": "ok"}


def test_proxy_load_and_assert_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    import trend_analysis.llm_proxy.server as srv

    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.FastAPI = object  # type: ignore[attr-defined]
    fake_fastapi.HTTPException = object  # type: ignore[attr-defined]
    fake_fastapi.Request = object  # type: ignore[attr-defined]
    fake_resp = types.ModuleType("fastapi.responses")
    fake_resp.Response = object  # type: ignore[attr-defined]
    fakes = {
        "fastapi": fake_fastapi,
        "fastapi.responses": fake_resp,
        "httpx": types.ModuleType("httpx"),
        "uvicorn": types.ModuleType("uvicorn"),
    }
    monkeypatch.setattr(srv.importlib, "import_module", lambda n: fakes[n])
    # _load_deps rebinds these module globals; register them first so the test leaves them as it found them
    for name in ("FastAPI", "HTTPException", "Request", "Response", "httpx", "uvicorn"):
        monkeypatch.setattr(srv, name, getattr(srv, name, None), raising=False)
    monkeypatch.setattr(srv, "_DEPS_AVAILABLE", False, raising=False)
    srv._load_deps()
    assert srv._DEPS_AVAILABLE is True and srv.FastAPI is fake_fastapi.FastAPI
    srv._assert_deps()
    monkeypatch.setattr(
        srv.importlib, "import_module", lambda n: (_ for _ in ()).throw(ImportError("no"))
    )
    monkeypatch.setattr(srv, "_DEPS_AVAILABLE", False, raising=False)
    srv._load_deps()
    assert srv.FastAPI is None and srv._DEPS_AVAILABLE is False
    with pytest.raises(RuntimeError, match="Required dependencies not available"):
        srv._assert_deps()


def test_proxy_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    import trend_analysis.llm_proxy.server as srv

    for key in (
        "TS_STREAMLIT_API_KEY",
        "TS_OPENAI_STREAMLIT",
        "OPENAI_API_KEY",
        "TREND_LLM_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    assert srv._resolve_upstream_key() is None
    monkeypatch.setenv("TREND_LLM_API_KEY", "k-last")
    monkeypatch.setenv("OPENAI_API_KEY", "k-open")
    assert srv._resolve_upstream_key() == "k-open"
    assert srv._filter_response_headers({"Content-Length": "3", "X-A": "1", "TE": "x"}) == {
        "X-A": "1"
    }


def _req(
    headers: dict[str, str] | None = None, body: bytes = b"{}", query: Any = ""
) -> SimpleNamespace:
    async def _body() -> bytes:
        return body

    return SimpleNamespace(
        headers=headers or {}, url=SimpleNamespace(query=query), method="POST", body=_body
    )


def test_proxy_handle_request_auth_and_success(monkeypatch: pytest.MonkeyPatch) -> None:
    mods, state = _proxy_fakes()
    srv = _fake_proxy(monkeypatch, mods)
    monkeypatch.delenv("TS_STREAMLIT_API_KEY", raising=False)
    monkeypatch.delenv("TS_OPENAI_STREAMLIT", raising=False)
    monkeypatch.setenv("TS_LLM_PROXY_TOKEN", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "upkey")
    proxy = srv.LLMProxy("https://up.example")
    forward = state["routes"][("FWD", "/v1/{path:path}")]
    resp_fwd = asyncio.run(forward("chat/completions", _req({"authorization": "Bearer secret"})))
    assert resp_fwd.status_code == 201 and resp_fwd.content == b"OK"
    with pytest.raises(mods["HTTPException"]) as e1:  # type: ignore[attr-defined]
        asyncio.run(proxy._handle_request(_req(), "chat"))
    assert (e1.value.status_code, e1.value.detail) == (401, "Missing proxy token")
    with pytest.raises(mods["HTTPException"]) as e2:  # type: ignore[attr-defined]
        asyncio.run(proxy._handle_request(_req({"authorization": "Bearer wrong"}), "chat"))
    assert (e2.value.status_code, e2.value.detail) == (403, "Invalid proxy token")
    resp = asyncio.run(
        proxy._handle_request(
            _req({"authorization": "Bearer secret", "host": "h"}, query=b"a=1"), "chat"
        )
    )
    assert resp.status_code == 201 and resp.content == b"OK" and resp.headers == {"X-Keep": "yes"}
    assert state["req"][1] == "https://up.example/v1/chat?a=1"
    assert state["req"][2]["authorization"] == "Bearer upkey"


def test_proxy_handle_request_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    mods, state = _proxy_fakes()
    srv = _fake_proxy(monkeypatch, mods)
    monkeypatch.delenv("TS_LLM_PROXY_TOKEN", raising=False)
    for key in (
        "TS_STREAMLIT_API_KEY",
        "TS_OPENAI_STREAMLIT",
        "OPENAI_API_KEY",
        "TREND_LLM_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    proxy = srv.LLMProxy("https://up.example")
    with pytest.raises(mods["HTTPException"]) as e0:  # type: ignore[attr-defined]
        asyncio.run(proxy._handle_request(_req(), "chat"))
    assert (e0.value.status_code, e0.value.detail) == (500, "Upstream API key not configured")
    monkeypatch.setenv("OPENAI_API_KEY", "upkey")
    monkeypatch.setenv("TS_LLM_PROXY_MAX_BODY_BYTES", "2")
    with pytest.raises(mods["HTTPException"]) as e1:  # type: ignore[attr-defined]
        asyncio.run(proxy._handle_request(_req(body=b"12345"), "chat"))
    assert e1.value.status_code == 413
    monkeypatch.delenv("TS_LLM_PROXY_MAX_BODY_BYTES")

    async def raise_timeout(**k: Any) -> Any:
        raise _FakeTimeout("slow")

    async def raise_req(**k: Any) -> Any:
        raise _FakeReqErr("down")

    async def raise_http(**k: Any) -> Any:
        raise _FakeHTTPError("bad")

    for fn, code, detail in (
        (raise_timeout, 504, "Upstream service timed out"),
        (raise_req, 502, "Error contacting upstream service"),
        (raise_http, 502, "Upstream service error"),
    ):
        proxy.client.request = fn  # type: ignore[method-assign]
        with pytest.raises(mods["HTTPException"]) as ei:  # type: ignore[attr-defined]
            asyncio.run(proxy._handle_request(_req(), "chat"))
        assert (ei.value.status_code, ei.value.detail) == (code, detail)


def test_proxy_start_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    mods, state = _proxy_fakes()
    srv = _fake_proxy(monkeypatch, mods)
    monkeypatch.delenv("TS_LLM_PROXY_TOKEN", raising=False)
    seen: dict[str, Any] = {}

    class Cfg:
        def __init__(self, app: Any, host: str, port: int, log_level: str) -> None:
            seen.update(host=host, port=port)

    class Server:
        def __init__(self, config: Any) -> None:
            pass

        async def serve(self) -> None:
            seen["served"] = True

    mods["uvicorn"] = SimpleNamespace(Config=Cfg, Server=Server)
    monkeypatch.setattr(srv, "uvicorn", mods["uvicorn"], raising=False)
    proxy = srv.LLMProxy("https://up.example")
    asyncio.run(proxy.start(host="0.0.0.0", port=9999))
    assert seen == {"host": "0.0.0.0", "port": 9999, "served": True}
    monkeypatch.setattr(srv, "uvicorn", None, raising=False)
    with pytest.raises(RuntimeError, match="uvicorn dependency is required"):
        asyncio.run(proxy.start())
    asyncio.run(proxy.close())
    assert state["closed"] is True
