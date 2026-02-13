from __future__ import annotations

import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

import trend_analysis.monte_carlo.runner as runner_module
from trend_analysis import cli
from trend_analysis.api import RunResult
from trend_analysis.monte_carlo.results import MonteCarloResults
from trend_analysis.monte_carlo.scenario import MonteCarloScenario, MonteCarloSettings


def _write_scenario(path: Path, *, name: str = "mc_test", extra: str = "") -> None:
    payload = f"""
scenario:
  name: {name}
  description: Test scenario

base_config: config/defaults.yml

monte_carlo:
  mode: two_layer
  n_paths: 1
  horizon_years: 1.0
  frequency: M
  seed: 7
"""
    if extra:
        payload = payload + "\n" + extra
    path.write_text(payload.strip() + "\n", encoding="utf-8")


def _write_prices(path: Path) -> None:
    payload = "\n".join(
        [
            "Date,AssetA,AssetB",
            "2020-01-31,100,101",
            "2020-02-29,102,103",
            "2020-03-31,104,105",
            "2020-04-30,106,107",
        ]
    )
    path.write_text(payload + "\n", encoding="utf-8")


def _write_registry(path: Path) -> None:
    registry = """
scenarios:
  - name: alpha
    description: Alpha scenario
    tags: [alpha, core]
    path: alpha.yml
  - name: beta
    description: Beta scenario
    tags: [beta]
    path: beta.yml
"""
    path.write_text(registry.strip() + "\n", encoding="utf-8")
    for scenario in ("alpha.yml", "beta.yml"):
        (path.parent / scenario).write_text("", encoding="utf-8")


def test_mc_validate_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scenario_path = tmp_path / "scenario.yml"
    _write_scenario(scenario_path)

    rc = cli.main(["mc", "validate", str(scenario_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "mc_test" in out


def test_mc_validate_invalid(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scenario_path = tmp_path / "invalid.yml"
    scenario_path.write_text(
        """
scenario:
  name: broken
monte_carlo:
  mode: two_layer
  n_paths: 1
  horizon_years: 1.0
  frequency: M
""".strip() + "\n",
        encoding="utf-8",
    )

    rc = cli.main(["mc", "validate", str(scenario_path)])

    assert rc == 1
    err = capsys.readouterr().err
    assert "base_config" in err


def test_mc_run_overrides_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario_path = tmp_path / "scenario.yml"
    data_path = tmp_path / "prices.csv"
    output_dir = tmp_path / "bundle"
    _write_scenario(scenario_path)
    _write_prices(data_path)

    def fake_run_simulation(*_args, **_kwargs) -> RunResult:
        metrics = pd.DataFrame({"sharpe_ratio": [1.23]}, index=["equal_weight"])
        return RunResult(metrics=metrics, details={}, seed=0, environment={})

    monkeypatch.setattr(runner_module, "run_simulation", fake_run_simulation)

    rc = cli.main(
        [
            "mc",
            "run",
            "--scenario",
            str(scenario_path),
            "--data",
            str(data_path),
            "--out",
            str(output_dir),
            "--n-paths",
            "3",
            "--jobs",
            "2",
            "--seed",
            "99",
        ]
    )

    assert rc == 0

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["settings"]["n_paths"] == 3
    assert manifest["settings"]["seed"] == 99
    assert manifest["settings"]["jobs"] == 2

    assert (output_dir / "results.csv").exists()
    assert (output_dir / "summary.csv").exists()


def test_mc_run_runtime_error_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scenario_path = tmp_path / "scenario.yml"
    data_path = tmp_path / "prices.csv"
    _write_scenario(scenario_path)
    _write_prices(data_path)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(runner_module.MonteCarloRunner, "run", _boom)

    rc = cli.main(
        [
            "mc",
            "run",
            "--scenario",
            str(scenario_path),
            "--data",
            str(data_path),
        ]
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "boom" in err


def test_mc_run_rejects_invalid_format_overrides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scenario_path = tmp_path / "scenario.yml"
    _write_scenario(scenario_path)

    rc = cli.main(["mc", "run", "--scenario", str(scenario_path), "--formats", "xml"])

    assert rc == 1
    err = capsys.readouterr().err
    assert "format overrides contains unsupported values: xml" in err


def test_mc_manifest_includes_required_keys_and_uses_utc_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = MonteCarloSettings(
        mode="two_layer",
        n_paths=5,
        horizon_years=1.0,
        frequency="M",
        seed=12,
        jobs=2,
    )
    scenario = MonteCarloScenario(
        name="alpha",
        description="Alpha scenario",
        version="1.0",
        base_config=tmp_path / "config.yml",
        monte_carlo=settings,
    )
    results_frame = pd.DataFrame({"path_id": [1], "strategy": ["eq"]})
    summary_frame = pd.DataFrame({"strategy": ["eq"], "paths": [1]})
    results = MonteCarloResults(
        mode="two_layer",
        evaluations=[],
        errors=[],
        results_frame=results_frame,
        summary_frame=summary_frame,
    )

    captured: dict[str, object] = {}
    fixed_time = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            captured["tz"] = tz
            return fixed_time if tz is not None else fixed_time.replace(tzinfo=None)

    monkeypatch.setattr(cli, "datetime", FixedDateTime)

    output_dir = tmp_path / "manifest"
    manifest_path = cli._write_mc_manifest(
        output_dir,
        scenario=scenario,
        results=results,
        overrides={},
        exported_files={},
        data_path=None,
        jobs_used=4,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings_payload = manifest["settings"]
    assert {"n_paths", "jobs", "seed"} <= set(settings_payload)
    assert settings_payload["n_paths"] == 5
    assert settings_payload["seed"] == 12
    assert settings_payload["jobs"] == 4
    assert captured["tz"] is timezone.utc
    assert manifest["created_at"] == fixed_time.isoformat()


def test_mc_run_shows_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scenario_path = tmp_path / "scenario.yml"
    data_path = tmp_path / "prices.csv"
    output_dir = tmp_path / "bundle"
    _write_scenario(scenario_path)
    _write_prices(data_path)

    monkeypatch.setitem(sys.modules, "tqdm", ModuleType("tqdm"))

    def _fake_run(self, progress_callback=None, jobs=None):  # type: ignore[no-untyped-def]
        if progress_callback is not None:
            progress_callback({"completed": 1, "total": 1})
        results_frame = pd.DataFrame({"path_id": [1], "strategy": ["eq"]})
        summary_frame = pd.DataFrame({"strategy": ["eq"], "paths": [1]})
        return MonteCarloResults(
            mode="two_layer",
            evaluations=[],
            errors=[],
            results_frame=results_frame,
            summary_frame=summary_frame,
        )

    monkeypatch.setattr(runner_module.MonteCarloRunner, "run", _fake_run)

    rc = cli.main(
        [
            "mc",
            "run",
            "--scenario",
            str(scenario_path),
            "--data",
            str(data_path),
            "--out",
            str(output_dir),
        ]
    )

    assert rc == 0
    err = capsys.readouterr().err
    assert "Progress: 1/1" in err


def test_mc_run_uses_tqdm_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scenario_path = tmp_path / "scenario.yml"
    data_path = tmp_path / "prices.csv"
    output_dir = tmp_path / "bundle"
    _write_scenario(scenario_path)
    _write_prices(data_path)

    class _DummyTqdm:
        def __init__(self) -> None:
            self.total = 0
            self.updated = 0
            self.refreshed = 0
            self.closed = 0

        def update(self, value: int) -> None:
            self.updated += value

        def refresh(self) -> None:
            self.refreshed += 1

        def close(self) -> None:
            self.closed += 1

    dummy = _DummyTqdm()
    dummy.total = 1
    tqdm_module = ModuleType("tqdm")
    tqdm_module.tqdm = dummy
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)

    def _fake_run(self, progress_callback=None, jobs=None):  # type: ignore[no-untyped-def]
        if progress_callback is not None:
            progress_callback({"completed": 1, "total": 1})
        results_frame = pd.DataFrame({"path_id": [1], "strategy": ["eq"]})
        summary_frame = pd.DataFrame({"strategy": ["eq"], "paths": [1]})
        return MonteCarloResults(
            mode="two_layer",
            evaluations=[],
            errors=[],
            results_frame=results_frame,
            summary_frame=summary_frame,
        )

    monkeypatch.setattr(runner_module.MonteCarloRunner, "run", _fake_run)

    rc = cli.main(
        [
            "mc",
            "run",
            "--scenario",
            str(scenario_path),
            "--data",
            str(data_path),
            "--out",
            str(output_dir),
        ]
    )

    assert rc == 0
    err = capsys.readouterr().err
    assert "Progress: " not in err
    assert dummy.updated == 1
    assert dummy.closed == 1


def test_is_valid_tqdm_instance_requires_callable_methods() -> None:
    class _DummyTqdm:
        def __init__(self) -> None:
            self.total = 1
            self.update = 1

        def refresh(self) -> None:
            return None

        def close(self) -> None:
            return None

    assert cli._is_valid_tqdm_instance(_DummyTqdm()) is False


def test_mc_run_reconfigures_tqdm_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scenario_path = tmp_path / "scenario.yml"
    data_path = tmp_path / "prices.csv"
    output_dir = tmp_path / "bundle"
    _write_scenario(scenario_path)
    _write_prices(data_path)

    class _DummyTqdm:
        def __init__(self) -> None:
            self.total = 0
            self.unit = "items"
            self.updated = 0
            self.refreshed = 0
            self.closed = 0

        def update(self, value: int) -> None:
            self.updated += value

        def refresh(self) -> None:
            self.refreshed += 1

        def close(self) -> None:
            self.closed += 1

    dummy = _DummyTqdm()
    tqdm_module = ModuleType("tqdm")
    tqdm_module.tqdm = dummy
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)

    def _fake_run(self, progress_callback=None, jobs=None):  # type: ignore[no-untyped-def]
        if progress_callback is not None:
            progress_callback({"completed": 1, "total": 1})
        results_frame = pd.DataFrame({"path_id": [1], "strategy": ["eq"]})
        summary_frame = pd.DataFrame({"strategy": ["eq"], "paths": [1]})
        return MonteCarloResults(
            mode="two_layer",
            evaluations=[],
            errors=[],
            results_frame=results_frame,
            summary_frame=summary_frame,
        )

    monkeypatch.setattr(runner_module.MonteCarloRunner, "run", _fake_run)

    rc = cli.main(
        [
            "mc",
            "run",
            "--scenario",
            str(scenario_path),
            "--data",
            str(data_path),
            "--out",
            str(output_dir),
        ]
    )

    assert rc == 0
    err = capsys.readouterr().err
    assert "Progress: " not in err
    assert dummy.total == 1
    assert dummy.unit == "path"
    assert dummy.updated == 1
    assert dummy.closed == 1


def test_mc_run_dry_run_skips_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scenario_path = tmp_path / "scenario.yml"
    _write_scenario(scenario_path)

    called = {"run": False}

    def _run(self, progress_callback=None):  # type: ignore[no-untyped-def]
        called["run"] = True
        results_frame = pd.DataFrame({"path_id": [1], "strategy": ["eq"]})
        summary_frame = pd.DataFrame({"strategy": ["eq"], "paths": [1]})
        return MonteCarloResults(
            mode="two_layer",
            evaluations=[],
            errors=[],
            results_frame=results_frame,
            summary_frame=summary_frame,
        )

    monkeypatch.setattr(runner_module.MonteCarloRunner, "run", _run)

    rc = cli.main(
        [
            "mc",
            "run",
            "--scenario",
            str(scenario_path),
            "--dry-run",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Dry run complete" in out
    assert called["run"] is False


def test_mc_list_registry_and_tags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    registry_path = tmp_path / "index.yml"
    _write_registry(registry_path)

    rc = cli.main(["mc", "list", "--registry", str(registry_path), "--format", "json"])

    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert {entry["name"] for entry in payload} == {"alpha", "beta"}

    rc = cli.main(
        [
            "mc",
            "list",
            "--registry",
            str(registry_path),
            "--tags",
            "alpha",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert [entry["name"] for entry in payload] == ["alpha"]

    rc = cli.main(
        [
            "mc",
            "list",
            "--registry",
            str(registry_path),
            "--tags",
            "alpha",
            "--tags",
            "beta",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert {entry["name"] for entry in payload} == {"alpha", "beta"}

    rc = cli.main(
        [
            "mc",
            "list",
            "--registry",
            str(registry_path),
            "--tags",
            "alpha,beta",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert {entry["name"] for entry in payload} == {"alpha", "beta"}

    rc = cli.main(
        [
            "mc",
            "list",
            "--registry",
            str(registry_path),
            "--tags",
            " Alpha , BETA ",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert {entry["name"] for entry in payload} == {"alpha", "beta"}


def test_mc_list_table_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    registry_path = tmp_path / "index.yml"
    _write_registry(registry_path)

    rc = cli.main(["mc", "list", "--registry", str(registry_path)])

    assert rc == 0
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert lines
    header = lines[0]
    assert "Name" in header
    assert "Tags" in header
    assert "Description" in header
    assert "Path" in header
    assert "alpha" in out
    assert "beta" in out


def test_mc_list_missing_registry_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_registry = tmp_path / "missing.yml"

    rc = cli.main(["mc", "list", "--registry", str(missing_registry)])

    assert rc == 1
    err = capsys.readouterr().err
    assert "Failed to list Monte Carlo scenarios" in err


def test_mc_list_empty_registry_shows_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry_path = tmp_path / "index.yml"
    registry_path.write_text("scenarios: []\n", encoding="utf-8")

    rc = cli.main(["mc", "list", "--registry", str(registry_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "No Monte Carlo scenarios found." in out


def test_mc_viz_requires_at_least_one_output_flag(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["mc", "viz", "--bundle", "bundle_dir", "--out", "export_dir"])

    assert rc == 1
    err = capsys.readouterr().err
    assert "requires at least one output flag" in err


def test_mc_viz_loads_summary_and_results_frames(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    pd.DataFrame({"metric": ["cagr"], "value": [0.12]}).to_csv(
        bundle_dir / "summary.csv", index=False
    )
    pd.DataFrame({"path_id": [1, 2], "terminal_nav": [112.0, 98.4]}).to_csv(
        bundle_dir / "results.csv", index=False
    )

    rc = cli.main(
        [
            "mc",
            "viz",
            "--bundle",
            str(bundle_dir),
            "--out",
            str(tmp_path / "exports"),
            "--charts",
            "fan,risk_return",
            "--html",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Loaded MC bundle frames" in out
    assert "summary_rows=1" in out
    assert "results_rows=2" in out
    assert (tmp_path / "exports" / "plots").is_dir()


def test_mc_viz_errors_when_summary_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    pd.DataFrame({"path_id": [1], "terminal_nav": [101.0]}).to_csv(
        bundle_dir / "results.csv", index=False
    )

    rc = cli.main(
        [
            "mc",
            "viz",
            "--bundle",
            str(bundle_dir),
            "--out",
            str(tmp_path / "exports"),
            "--json",
        ]
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert "Missing required MC summary file" in err


def test_mc_viz_errors_when_results_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    pd.DataFrame({"metric": ["vol"], "value": [0.09]}).to_csv(
        bundle_dir / "summary.csv", index=False
    )

    rc = cli.main(
        [
            "mc",
            "viz",
            "--bundle",
            str(bundle_dir),
            "--out",
            str(tmp_path / "exports"),
            "--html",
        ]
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert "Missing required MC results file" in err


def test_mc_viz_errors_when_multiple_required_inputs_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    rc = cli.main(
        [
            "mc",
            "viz",
            "--bundle",
            str(bundle_dir),
            "--out",
            str(tmp_path / "exports"),
            "--json",
        ]
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert "Missing required MC input files" in err
    assert "summary" in err
    assert "results" in err


def test_mc_viz_errors_when_results_file_is_corrupted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    pd.DataFrame({"metric": ["vol"], "value": [0.09]}).to_csv(
        bundle_dir / "summary.csv", index=False
    )
    (bundle_dir / "results.json").write_text("{bad json", encoding="utf-8")

    rc = cli.main(
        [
            "mc",
            "viz",
            "--bundle",
            str(bundle_dir),
            "--out",
            str(tmp_path / "exports"),
            "--html",
        ]
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert "Failed to read results data" in err


def test_mc_viz_loads_optional_nav_paths_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    pd.DataFrame({"metric": ["cagr"], "value": [0.12]}).to_csv(
        bundle_dir / "summary.csv", index=False
    )
    pd.DataFrame({"path_id": [1, 2], "terminal_nav": [112.0, 98.4]}).to_csv(
        bundle_dir / "results.csv", index=False
    )
    (bundle_dir / "nav_paths.parquet").write_text("placeholder", encoding="utf-8")

    import trend.mc.io as _mc_io

    monkeypatch.setattr(
        _mc_io.pd, "read_parquet", lambda _path: pd.DataFrame({"path_id": [1, 2, 3]})
    )

    rc = cli.main(
        [
            "mc",
            "viz",
            "--bundle",
            str(bundle_dir),
            "--out",
            str(tmp_path / "exports"),
            "--charts",
            "fan,risk_return",
            "--html",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "nav_paths_rows=3" in out


def test_mc_viz_errors_when_chart_identifier_is_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    pd.DataFrame({"metric": ["cagr"], "value": [0.12]}).to_csv(
        bundle_dir / "summary.csv", index=False
    )
    pd.DataFrame({"path_id": [1, 2], "terminal_nav": [112.0, 98.4]}).to_csv(
        bundle_dir / "results.csv", index=False
    )

    rc = cli.main(
        [
            "mc",
            "viz",
            "--bundle",
            str(bundle_dir),
            "--out",
            str(tmp_path / "exports"),
            "--charts",
            "fan,unknown",
            "--html",
        ]
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert "Unsupported chart identifier" in err


def test_mc_viz_routes_selected_charts_and_exports_requested_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    pd.DataFrame({"metric": ["cagr"], "value": [0.12]}).to_csv(
        bundle_dir / "summary.csv", index=False
    )
    pd.DataFrame({"path_id": [1, 2], "terminal_nav": [112.0, 98.4]}).to_csv(
        bundle_dir / "results.csv", index=False
    )

    call_order: list[str] = []

    def _builder(name: str):
        def _inner(
            _summary: pd.DataFrame, _results: pd.DataFrame, _nav: pd.DataFrame | None
        ) -> object:
            call_order.append(name)
            return object()

        return _inner

    import trend.mc.viz as _mc_viz

    monkeypatch.setattr(_mc_viz, "check_png_dependency", lambda: True)

    monkeypatch.setattr(
        _mc_viz,
        "_mc_chart_builders",
        lambda: {
            "fan": _builder("fan"),
            "path_dist": _builder("path_dist"),
            "risk_return": _builder("risk_return"),
        },
    )

    captured: dict[str, object] = {}

    def fake_save(
        charts: dict[str, object],
        destination: Path | str | None = None,
        *,
        include_json: bool = True,
        include_html: bool = True,
        include_png: bool = False,
        warnings: list[str] | None = None,
        **_kwargs: object,
    ) -> Path:
        assert destination is not None
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        captured["charts"] = list(charts.keys())
        captured["include_json"] = include_json
        captured["include_html"] = include_html
        captured["include_png"] = include_png
        with zipfile.ZipFile(dest_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in charts:
                if include_json:
                    archive.writestr(f"{name}.json", "{}")
                if include_html:
                    archive.writestr(f"{name}.html", "<html></html>")
                if include_png:
                    archive.writestr(f"{name}.png", b"PNG")
        if warnings is not None:
            warnings.append("stub warning")
        return dest_path

    from trend_analysis.monte_carlo import export_bundle as mc_export_bundle

    monkeypatch.setattr(mc_export_bundle, "save", fake_save)

    out_dir = tmp_path / "exports"
    rc = cli.main(
        [
            "mc",
            "viz",
            "--bundle",
            str(bundle_dir),
            "--out",
            str(out_dir),
            "--charts",
            "risk_return,fan",
            "--json",
            "--png",
        ]
    )

    assert rc == 0
    assert call_order == ["risk_return", "fan"]
    assert captured["charts"] == ["risk_return", "fan"]
    assert captured["include_json"] is True
    assert captured["include_html"] is False
    assert captured["include_png"] is True
    plots_dir = out_dir / "plots"
    assert plots_dir.is_dir()
    assert (plots_dir / "risk_return.json").exists()
    assert (plots_dir / "fan.json").exists()
    assert (plots_dir / "risk_return.png").exists()
    assert (plots_dir / "fan.png").exists()
    assert not (plots_dir / "risk_return.html").exists()


def test_mc_viz_acceptance_command_writes_plots_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    pd.DataFrame({"metric": ["cagr"], "value": [0.12]}).to_csv(
        bundle_dir / "summary.csv", index=False
    )
    pd.DataFrame({"path_id": [1, 2], "terminal_nav": [112.0, 98.4]}).to_csv(
        bundle_dir / "results.csv", index=False
    )

    import trend.mc.viz as _mc_viz

    monkeypatch.setattr(_mc_viz, "check_png_dependency", lambda: True)

    monkeypatch.setattr(
        _mc_viz,
        "_mc_chart_builders",
        lambda: {
            "fan": lambda _summary, _results, _nav: object(),
            "path_dist": lambda _summary, _results, _nav: object(),
            "risk_return": lambda _summary, _results, _nav: object(),
        },
    )

    def fake_save(
        charts: dict[str, object],
        destination: Path | str | None = None,
        *,
        include_json: bool = True,
        include_html: bool = True,
        include_png: bool = False,
        warnings: list[str] | None = None,
        **_kwargs: object,
    ) -> Path:
        assert destination is not None
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in charts:
                if include_json:
                    archive.writestr(f"{name}.json", "{}")
                if include_html:
                    archive.writestr(f"{name}.html", "<html></html>")
                if include_png:
                    archive.writestr(f"{name}.png", b"PNG")
        if warnings is not None:
            warnings.append("stub warning")
        return dest_path

    from trend_analysis.monte_carlo import export_bundle as mc_export_bundle

    monkeypatch.setattr(mc_export_bundle, "save", fake_save)

    out_dir = tmp_path / "exports"
    rc = cli.main(
        [
            "mc",
            "viz",
            "--bundle",
            str(bundle_dir),
            "--out",
            str(out_dir),
            "--charts",
            "fan,risk_return",
            "--html",
            "--json",
            "--png",
        ]
    )

    assert rc == 0
    plots_dir = out_dir / "plots"
    assert plots_dir.is_dir()
    for chart in ("fan", "risk_return"):
        for ext in ("html", "json", "png"):
            assert (plots_dir / f"{chart}.{ext}").exists()
