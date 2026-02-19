import argparse
import json
import logging
import numbers
import os
import platform
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

from trend.config_schema import CoreConfigError, validate_core_config
from trend.diagnostics import DiagnosticPayload

from . import export, pipeline
from . import logging as run_logging
from .api import run_simulation
from .config import format_validation_messages, load_config, validate_config
from .config.coverage import (
    ConfigCoverageTracker,
    activate_config_coverage,
    deactivate_config_coverage,
    wrap_config_for_coverage,
)
from .config.schema_validation import load_config as load_config_yaml
from .config.ui_mapping import build_config_from_ui_state
from .constants import DEFAULT_OUTPUT_DIRECTORY, DEFAULT_OUTPUT_FORMATS
from .data import load_csv
from .diagnostics import coerce_pipeline_result
from .io.market_data import (
    MarketDataMode,
    MarketDataValidationError,
)
from .io.market_data import (
    load_market_data_csv as load_mc_market_data_csv,
)
from .io.market_data import (
    load_market_data_parquet as load_mc_market_data_parquet,
)
from .io.ui_ingest import inspect_ui_date_issues, load_ui_dataset
from .logging_setup import setup_logging
from .monte_carlo.registry import (
    ScenarioRegistryEntry,
    list_scenarios,
    load_scenario,
    load_scenario_from_path,
)
from .monte_carlo.results import MonteCarloResults, export_results
from .monte_carlo.runner import MonteCarloRunner
from .monte_carlo.scenario import MonteCarloScenario, MonteCarloSettings
from .perf.rolling_cache import set_cache_enabled
from .presets import apply_trend_preset, get_trend_preset, list_preset_slugs
from .reporting.portfolio_series import select_primary_portfolio_series
from .reporting.run_artifacts import write_run_artifacts
from .signal_presets import (
    TrendSpecPreset,
    get_trend_spec_preset,
    list_trend_spec_presets,
)
from .universe_catalog import NamedUniverse, load_universe

APP_PATH = Path(__file__).resolve().parents[2] / "streamlit_app" / "app.py"
LOCK_PATH = Path(__file__).resolve().parents[2] / "requirements.lock"


def load_market_data_csv(
    path: str,
    *,
    errors: str | None = None,
    include_date_column: bool | None = None,
    **kwargs: Any,
) -> pd.DataFrame | None:
    """Backward-compatible shim retaining the legacy symbol for tests and
    CLI."""

    effective_kwargs = dict(kwargs)
    effective_kwargs.setdefault("errors", errors if errors is not None else "raise")
    effective_kwargs.setdefault(
        "include_date_column",
        include_date_column if include_date_column is not None else True,
    )
    return load_csv(path, **effective_kwargs)


def _maybe_validate_config(
    cfg: Any,
    *,
    base_path: Path,
    config_path: Path | None = None,
) -> bool:
    """Return True when configuration validation passes or is skipped."""

    payload: dict[str, Any] | None = None
    used_raw = False
    if config_path is not None:
        try:
            payload = load_config_yaml(config_path)
            used_raw = True
        except Exception:
            payload = None

    if payload is None:
        if isinstance(cfg, Mapping):
            payload = dict(cfg)
        elif hasattr(cfg, "model_dump"):
            try:
                payload = cfg.model_dump(exclude_none=True, exclude_unset=True, mode="json")
            except TypeError:
                try:
                    payload = cfg.model_dump(exclude_none=True, exclude_unset=True)
                except TypeError:
                    payload = cfg.model_dump(exclude_none=True)
        elif hasattr(cfg, "__dict__"):
            payload = dict(cfg.__dict__)

    if not payload or "version" not in payload:
        return True

    result = validate_config(payload, base_path=base_path, skip_required_fields=True)
    if result.valid:
        return True

    if used_raw:
        ignored = []
        for issue in result.errors:
            if issue.path == "output":
                ignored.append(issue)
                continue
            if issue.path == "preprocessing.steps" and isinstance(issue.actual, list):
                ignored.append(issue)
                continue
        if ignored and len(ignored) == len(result.errors):
            return True

    print("Config validation failed:", file=sys.stderr)
    for line in format_validation_messages(result):
        print(f"- {line}", file=sys.stderr)
    return False


def _maybe_track_config_coverage(config_path: Path, input_path: str) -> bool:
    try:
        payload = load_config_yaml(config_path)
    except Exception:
        return True
    if not isinstance(payload, Mapping):
        return True

    data_section = dict(payload.get("data") or {})
    if input_path and not data_section.get("csv_path") and not data_section.get("managers_glob"):
        data_section["csv_path"] = input_path
        payload = dict(payload)
        payload["data"] = data_section

    try:
        validate_core_config(payload, base_path=config_path.parent)
    except CoreConfigError as exc:
        print(f"Config coverage validation failed: {exc}", file=sys.stderr)
        return False
    return True


def _report_pipeline_diagnostic(
    diagnostic: DiagnosticPayload,
    *,
    structured_log: bool,
    run_id: str,
) -> None:
    """Print and log a structured diagnostic emitted by the pipeline."""

    context = diagnostic.context or {}
    text = f"Pipeline skipped ({diagnostic.reason_code}): {diagnostic.message}"
    print(text)
    safe_fields = {k: v for k, v in context.items() if isinstance(k, str)}
    maybe_log_step(
        structured_log,
        run_id,
        "pipeline_diagnostic",
        diagnostic.message,
        reason_code=diagnostic.reason_code,
        **safe_fields,
    )


def _apply_trend_spec_preset(cfg: Any, preset: TrendSpecPreset) -> None:
    """Merge TrendSpec preset parameters into ``cfg`` in-place."""

    payload = preset.as_signal_config()
    if isinstance(cfg, dict):
        existing = cfg.get("signals")
        merged = dict(existing) if isinstance(existing, Mapping) else {}
        merged.update(payload)
        cfg["signals"] = merged
        cfg["trend_spec_preset"] = preset.name
        return

    existing = getattr(cfg, "signals", None)
    if isinstance(existing, Mapping):
        merged = dict(existing)
        merged.update(payload)
    else:
        merged = dict(payload)

    try:
        setattr(cfg, "signals", merged)
    except ValueError:
        object.__setattr__(cfg, "signals", merged)
    try:
        setattr(cfg, "trend_spec_preset", preset.name)
    except ValueError:
        object.__setattr__(cfg, "trend_spec_preset", preset.name)


def _log_step(run_id: str, event: str, message: str, level: str = "INFO", **fields: Any) -> None:
    """Internal indirection for structured logging.

    Tests monkeypatch this symbol directly (`_log_step`) rather than the public
    logging module function. Keeping this thin wrapper preserves the existing
    runtime behaviour while allowing tests to intercept calls without touching
    the logging subsystem.
    """
    run_logging.log_step(run_id, event, message, level=level, **fields)


def _extract_cache_stats(payload: object) -> dict[str, int] | None:
    """Return the most recent cache statistics embedded in ``payload``.

    Walks nested mappings and sequences looking for dictionaries that carry
    four integer fields: ``entries``, ``hits``, ``misses``, and ``incremental_updates``.
    These fields represent cache usage and performance counters during multi-period
    trend analysis, such as the number of cache entries, cache hits, cache misses,
    and incremental updates performed. The multi-period engine records a snapshot
    after every period, so the **last** occurrence reflects the final counters
    relevant to the analysis. Traversal intentionally skips pandas and NumPy
    containers to avoid expensive recursion through frames.
    """

    required = ("entries", "hits", "misses", "incremental_updates")
    found: list[dict[str, int]] = []

    def _visit(obj: object) -> None:
        if isinstance(obj, (pd.Series, pd.DataFrame, np.ndarray)):
            return
        if isinstance(obj, Mapping):
            if all(k in obj for k in required):
                candidate: dict[str, int] = {}
                for key in required:
                    value = obj.get(key)
                    if isinstance(value, numbers.Integral):
                        candidate[key] = int(value)
                    elif isinstance(value, numbers.Real) and float(value).is_integer():
                        candidate[key] = int(float(value))
                    else:
                        break
                else:
                    found.append(candidate)
            for value in obj.values():
                _visit(value)
            return
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            for item in obj:
                _visit(item)

    _visit(payload)
    return found[-1] if found else None


def _apply_universe_mask(df: pd.DataFrame, mask: pd.DataFrame, *, date_column: str) -> pd.DataFrame:
    """Apply a time-varying membership mask to returns data."""

    if mask.empty:
        return df
    working = df.copy()
    lookup = {str(col).lower(): col for col in working.columns}
    try:
        date_col = lookup[date_column.lower()]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise KeyError(f"Date column '{date_column}' is missing from the returns data") from exc

    working[date_col] = pd.to_datetime(working[date_col])
    working = working.set_index(date_col)
    aligned_mask = mask.reindex(index=working.index, fill_value=False)

    missing = [col for col in aligned_mask.columns if col not in working.columns]
    if missing:
        preview = ", ".join(missing[:3])
        raise KeyError(
            "Universe members missing from returns data: "
            f"{preview}" + ("…" if len(missing) > 3 else "")
        )

    masked = working.copy()
    masked.loc[:, aligned_mask.columns] = masked.loc[:, aligned_mask.columns].where(aligned_mask)
    masked.reset_index(inplace=True)
    return masked


def _attach_universe_paths(cfg: Any, spec: NamedUniverse, *, csv_path: str | None) -> None:
    """Persist the selected universe paths onto ``cfg.data`` when possible."""

    membership_value = str(spec.membership_path)
    csv_value = csv_path
    data_section = getattr(cfg, "data", None)
    if isinstance(data_section, Mapping):
        merged = dict(data_section)
        merged["universe_membership_path"] = membership_value
        if csv_value:
            merged.setdefault("csv_path", csv_value)
        try:
            setattr(cfg, "data", merged)
        except Exception:
            object.__setattr__(cfg, "data", merged)
        return

    if data_section is None:
        payload: dict[str, str] = {"universe_membership_path": membership_value}
        if csv_value:
            payload["csv_path"] = csv_value
        try:
            setattr(cfg, "data", payload)
        except Exception:
            object.__setattr__(cfg, "data", payload)
        return

    try:
        setattr(data_section, "universe_membership_path", membership_value)
    except Exception:
        try:
            object.__setattr__(data_section, "universe_membership_path", membership_value)
        except Exception:
            data_section = None

    if csv_value and data_section is not None:
        try:
            setattr(data_section, "csv_path", csv_value)
        except Exception:
            try:
                object.__setattr__(data_section, "csv_path", csv_value)
            except Exception:
                pass


def _resolve_artifact_paths(
    out_dir: Path, filename: str, keys: Sequence[str], formats: Sequence[str]
) -> list[Path]:
    """Return the expected artifact paths for ``filename`` and ``formats``."""

    seen: set[Path] = set()
    paths: list[Path] = []
    for fmt in formats:
        fmt_norm = (fmt or "").lower()
        fmt_norm = "xlsx" if fmt_norm == "excel" else fmt_norm
        if not fmt_norm:
            continue
        if fmt_norm == "xlsx":
            candidate = out_dir / f"{filename}.xlsx"
            if candidate not in seen:
                paths.append(candidate)
                seen.add(candidate)
            continue
        for key in keys:
            candidate = out_dir / f"{filename}_{key}.{fmt_norm}"
            if candidate in seen:
                continue
            paths.append(candidate)
            seen.add(candidate)
    return paths


def check_environment(lock_path: Path | None = None) -> int:
    """Print Python and package versions, reporting mismatches."""

    lock_file = lock_path or LOCK_PATH
    print(f"Python {platform.python_version()}")
    if not lock_file.exists():
        print(f"Lock file not found: {lock_file}")
        return 1

    mismatches: list[tuple[str, str | None, str]] = []
    for line in lock_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            continue
        name, expected = line.split("==", 1)
        name = name.strip()
        expected = expected.split()[0]
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        line_out = f"{name} {installed or 'not installed'} (expected {expected})"
        print(line_out)
        if installed != expected:
            mismatches.append((name, installed, expected))

    if mismatches:
        print("Mismatches detected:")
        for name, installed, expected in mismatches:
            print(f"- {name}: installed {installed or 'none'}, expected {expected}")
        return 1

    print("All packages match lockfile.")
    return 0


def maybe_log_step(enabled: bool, run_id: str, event: str, message: str, **fields: Any) -> None:
    """Log a structured step when ``enabled`` is True."""
    if enabled:
        _log_step(run_id, event, message, **fields)


def _load_ui_payload(path: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    payload_any: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_any, Mapping):
        raise ValueError("UI params must be a JSON object")
    payload = dict(payload_any)
    model_state = payload.get("model_state")
    if isinstance(model_state, Mapping):
        return payload, model_state
    return {"model_state": payload}, payload


def _looks_like_model_state(payload: Mapping[str, Any]) -> bool:
    ui_keys = {
        "lookback_periods",
        "evaluation_periods",
        "selection_count",
        "metric_weights",
        "trend_window",
        "trend_lag",
        "trend_min_periods",
        "trend_zscore",
        "trend_vol_adjust",
        "trend_vol_target",
        "vol_adjust_enabled",
        "risk_target",
        "multi_period_enabled",
        "multi_period_frequency",
        "start_date",
        "end_date",
        "date_mode",
    }
    config_keys = {
        "version",
        "data",
        "preprocessing",
        "vol_adjust",
        "sample_split",
        "portfolio",
        "metrics",
        "export",
        "run",
        "benchmarks",
        "regime",
        "robustness",
        "multi_period",
    }
    if any(key in payload for key in config_keys):
        return False
    hits = sum(1 for key in ui_keys if key in payload)
    return hits >= 3 or ("metric_weights" in payload and "selection_count" in payload)


def _should_handle_as_ui_config(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    try:
        payload_any: Any = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload_any, Mapping):
        return False
    if "model_state" in payload_any:
        return True
    return _looks_like_model_state(payload_any)


def _run_from_ui_payload(
    *,
    params_path: Path,
    data_path: Path,
    auto_fix_dates: bool,
    yes: bool,
    no_cache: bool,
    log_file: Path | None,
    structured_log: bool,
    bundle: Path | None,
) -> int:
    try:
        payload, model_state = _load_ui_payload(params_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if "risk_free_column" not in model_state:
        rf_column = payload.get("selected_risk_free")
        if isinstance(rf_column, str) and rf_column.strip():
            model_state = dict(model_state)
            model_state["risk_free_column"] = rf_column.strip()

    benchmark = payload.get("selected_benchmark")
    if not benchmark:
        info_benchmark = model_state.get("info_ratio_benchmark")
        if isinstance(info_benchmark, str) and info_benchmark.strip():
            benchmark = info_benchmark

    set_cache_enabled(not no_cache)

    if auto_fix_dates:
        try:
            issues = inspect_ui_date_issues(str(data_path))
        except MarketDataValidationError as exc:
            print(exc.user_message, file=sys.stderr)
            for issue in exc.issues:
                print(f"- {issue}", file=sys.stderr)
            return 1

        has_fixable = issues.has_corrections or issues.total_droppable_rows > 0
        if has_fixable and not yes:
            if not sys.stdin.isatty():
                print(
                    "Date corrections require confirmation. Re-run with --yes to approve.",
                    file=sys.stderr,
                )
                return 1
            prompt = (
                f"Apply {len(issues.corrections)} date correction(s) and "
                f"drop {issues.total_droppable_rows} row(s)? [y/N]: "
            )
            response = input(prompt).strip().lower()
            if response not in {"y", "yes"}:
                print("Cancelled date corrections.")
                return 1

    try:
        returns, meta, summary = load_ui_dataset(
            str(data_path),
            auto_fix_dates=auto_fix_dates,
        )
    except MarketDataValidationError as exc:
        print(exc.user_message, file=sys.stderr)
        for issue in exc.issues:
            print(f"- {issue}", file=sys.stderr)
        return 1

    if summary.corrected_dates or summary.dropped_rows:
        parts: list[str] = []
        if summary.corrected_dates:
            parts.append(f"{summary.corrected_dates} date correction(s)")
        if summary.dropped_rows:
            parts.append(f"{summary.dropped_rows} row(s) dropped")
        print(f"Applied UI-style date fixes: {', '.join(parts)}")

    csv_path = str(data_path) if data_path.suffix.lower() == ".csv" else None
    cfg = build_config_from_ui_state(
        returns=returns,
        model_state=model_state,
        benchmark=benchmark,
        frequency=meta.frequency,
        csv_path=csv_path,
    )

    returns_df = returns.reset_index()

    return _execute_analysis_run(
        cfg,
        returns_df,
        config_path=params_path,
        input_path=data_path,
        log_file=log_file,
        structured_log=structured_log,
        bundle=bundle,
    )


def _execute_analysis_run(
    cfg: Any,
    df: pd.DataFrame,
    *,
    config_path: Path | None,
    input_path: Path | None,
    log_file: Path | None,
    structured_log: bool,
    bundle: Path | None,
) -> int:
    import uuid

    split = cfg.sample_split
    required_keys = {"in_start", "in_end", "out_start", "out_end"}

    run_id = getattr(cfg, "run_id", None) or uuid.uuid4().hex[:12]
    try:
        setattr(cfg, "run_id", run_id)
    except Exception:
        pass
    log_path = log_file if log_file else run_logging.get_default_log_path(run_id)
    if structured_log:
        run_logging.init_run_logger(run_id, log_path)
    maybe_log_step(
        structured_log,
        run_id,
        "start",
        "CLI run initialised",
        config_path=str(config_path) if config_path else None,
    )

    res: Any = None
    pipeline_diagnostic: DiagnosticPayload | None = None
    if required_keys.issubset(split):
        maybe_log_step(
            structured_log,
            run_id,
            "load_data",
            "Loaded returns dataframe",
            rows=len(df),
        )
        run_result = run_simulation(cfg, df)
        maybe_log_step(
            structured_log,
            run_id,
            "pipeline_complete",
            "Pipeline execution finished",
            metrics_rows=len(run_result.metrics),
        )
        metrics_df = run_result.metrics
        res = run_result.details
        run_seed = run_result.seed
        pipeline_diagnostic = getattr(run_result, "diagnostic", None)
        if pipeline_diagnostic and not res:
            _report_pipeline_diagnostic(
                pipeline_diagnostic,
                structured_log=structured_log,
                run_id=run_id,
            )
            print("No results")
            return 0
        if isinstance(res, dict):
            port_ser = select_primary_portfolio_series(res, prefer_raw=True)
            if port_ser is not None:
                setattr(run_result, "portfolio", port_ser)
            bench_map = res.get("benchmarks") if isinstance(res, dict) else None
            if isinstance(bench_map, dict) and bench_map:
                first_bench = next(iter(bench_map.values()))
                setattr(run_result, "benchmark", first_bench)
            weights_user = res.get("weights_user_weight") if isinstance(res, dict) else None
            if weights_user is not None:
                setattr(run_result, "weights", weights_user)
    else:  # pragma: no cover - legacy fallback
        metrics_df = pipeline.run(cfg)
        full_result = pipeline.run_full(cfg)
        res, diag_payload = coerce_pipeline_result(full_result)
        run_seed = getattr(cfg, "seed", 42)
        pipeline_diagnostic = diag_payload or cast(
            DiagnosticPayload | None, metrics_df.attrs.get("diagnostic")
        )
        if res is None:
            res = {}
    if not res:
        if pipeline_diagnostic:
            _report_pipeline_diagnostic(
                pipeline_diagnostic,
                structured_log=structured_log,
                run_id=run_id,
            )
        print("No results")
        return 0

    text = export.format_summary_text(
        res,
        str(split.get("in_start")),
        str(split.get("in_end")),
        str(split.get("out_start")),
        str(split.get("out_end")),
    )
    print(text)
    maybe_log_step(structured_log, run_id, "summary_render", "Printed summary text")

    cache_stats = _extract_cache_stats(res)
    if cache_stats:
        print("\nCache statistics:")
        print(f"  Entries: {cache_stats['entries']}")
        print(f"  Hits: {cache_stats['hits']}")
        print(f"  Misses: {cache_stats['misses']}")
        print(f"  Incremental updates: {cache_stats['incremental_updates']}")
        maybe_log_step(
            structured_log,
            run_id,
            "cache_stats",
            "Cache statistics summary",
            **cache_stats,
        )

    export_cfg = cfg.export
    out_dir = export_cfg.get("directory")
    out_formats = export_cfg.get("formats")
    filename = export_cfg.get("filename", "analysis")
    if not out_dir and not out_formats:
        out_dir = DEFAULT_OUTPUT_DIRECTORY
        out_formats = DEFAULT_OUTPUT_FORMATS
    manifest_dir: Path | None = None
    if out_dir and out_formats:
        out_dir_path = Path(out_dir)
        fmt_list = list(out_formats)
        data = {"metrics": metrics_df}
        if isinstance(res, Mapping):
            export.append_narrative_section(data, res, config=cfg)
        maybe_log_step(
            structured_log,
            run_id,
            "export_start",
            "Beginning export",
            formats=fmt_list,
        )
        excel_requested = any(f.lower() in {"excel", "xlsx"} for f in fmt_list)
        if excel_requested:
            sheet_formatter = export.make_summary_formatter(
                res,
                str(split.get("in_start")),
                str(split.get("in_end")),
                str(split.get("out_start")),
                str(split.get("out_end")),
            )
            data["summary"] = export.summary_frame_from_result(res)
            export.export_to_excel(
                data,
                str(out_dir_path / f"{filename}.xlsx"),
                default_sheet_formatter=sheet_formatter,
            )
            other = [f for f in fmt_list if f.lower() not in {"excel", "xlsx"}]
            target_formats = other if other else fmt_list
        else:
            target_formats = fmt_list
        export.export_data(
            data,
            str(out_dir_path / filename),
            formats=target_formats,
        )
        data_keys = list(data.keys())
        artifact_paths = _resolve_artifact_paths(out_dir_path, filename, data_keys, fmt_list)
        maybe_log_step(
            structured_log,
            run_id,
            "export_complete",
            "Export finished",
        )
        config_payload: Any
        if hasattr(cfg, "model_dump"):
            try:
                config_payload = cfg.model_dump()
            except TypeError:  # pragma: no cover - defensive for exotic configs
                config_payload = cfg.model_dump()
        elif hasattr(cfg, "__dict__"):
            config_payload = dict(getattr(cfg, "__dict__"))
        else:
            config_payload = cfg
        if config_path is not None and input_path is not None:
            try:
                manifest_dir = write_run_artifacts(
                    output_dir=out_dir_path,
                    run_id=run_id,
                    config=config_payload,
                    config_path=str(config_path),
                    input_path=input_path,
                    data_frame=df,
                    metrics_frame=metrics_df,
                    run_details=res if isinstance(res, Mapping) else {},
                    exported_files=artifact_paths,
                    summary_text=text,
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                logging.getLogger(__name__).warning("Failed to write run artifacts: %s", exc)
            else:
                maybe_log_step(
                    structured_log,
                    run_id,
                    "run_artifacts",
                    "Run manifest written",
                    directory=str(manifest_dir),
                )

    if bundle:
        from .api import RunResult as _RR
        from .export.bundle import export_bundle

        bundle_path = Path(bundle)
        if bundle_path.is_dir():
            bundle_path = bundle_path / "analysis_bundle.zip"
        rr = locals().get("run_result")
        if rr is None:
            env = {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
            }
            rr = _RR(metrics_df, res, run_seed, env)
        setattr(rr, "config", getattr(cfg, "__dict__", {}))
        if input_path is not None:
            setattr(rr, "input_path", input_path)
        export_bundle(rr, bundle_path)
        print(f"Bundle written: {bundle_path}")
        maybe_log_step(
            structured_log,
            run_id,
            "bundle_complete",
            "Reproducibility bundle written",
            bundle=str(bundle_path),
        )
    maybe_log_step(
        structured_log,
        run_id,
        "end",
        "CLI run complete",
        log_file=str(log_path),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``trend-model`` command."""

    parser = argparse.ArgumentParser(prog="trend-model")
    parser.add_argument("--check", action="store_true", help="Print environment info and exit")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("gui", help="Launch Streamlit interface")

    run_p = sub.add_parser("run", help="Run analysis pipeline")
    run_p.add_argument("-c", "--config", required=True, help="Path to YAML config")
    run_p.add_argument("-i", "--input", required=True, help="Path to returns CSV")
    run_p.add_argument("--seed", type=int, help="Override random seed (takes precedence)")
    run_p.add_argument(
        "--bundle",
        nargs="?",
        const="analysis_bundle.zip",
        help="Write reproducibility bundle (optional path or default analysis_bundle.zip)",
    )
    run_p.add_argument(
        "--log-file",
        help="Path to JSONL structured log (defaults to outputs/logs/run_<id>.jsonl)",
    )
    run_p.add_argument(
        "--universe",
        help="Select a named universe defined under config/universe",
    )
    run_p.add_argument(
        "--no-structured-log",
        action="store_true",
        help="Disable structured JSONL logging for this run",
    )
    run_p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable persistent caching for rolling computations",
    )
    run_p.add_argument(
        "--preset",
        help="Apply a named trend preset to signal generation",
    )
    run_p.add_argument(
        "--config-coverage",
        action="store_true",
        help="Report which config keys were validated vs read",
    )
    run_p.add_argument(
        "--auto-fix-dates",
        action="store_true",
        help="Apply Streamlit-style date corrections automatically",
    )
    run_p.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation for date corrections",
    )

    run_ui_p = sub.add_parser(
        "run-ui",
        help="Deprecated: use 'run' with Streamlit JSON params",
    )
    run_ui_p.add_argument("--params", required=True, help="Path to Streamlit JSON params")
    run_ui_p.add_argument("--data", required=True, help="Path to returns CSV or Excel")
    run_ui_p.add_argument(
        "--auto-fix-dates",
        action="store_true",
        help="Apply Streamlit-style date corrections automatically",
    )
    run_ui_p.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation for date corrections",
    )
    run_ui_p.add_argument(
        "--bundle",
        nargs="?",
        const="analysis_bundle.zip",
        help="Write reproducibility bundle (optional path or default analysis_bundle.zip)",
    )
    run_ui_p.add_argument(
        "--log-file",
        help="Path to JSONL structured log (defaults to outputs/logs/run_<id>.jsonl)",
    )
    run_ui_p.add_argument(
        "--no-structured-log",
        action="store_true",
        help="Disable structured JSONL logging for this run",
    )
    run_ui_p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable persistent caching for rolling computations",
    )

    mc_p = sub.add_parser("mc", help="Monte Carlo scenario workflows")
    mc_sub = mc_p.add_subparsers(dest="mc_command", required=True)
    mc_list_p = mc_sub.add_parser("list", help="List registered Monte Carlo scenarios")
    mc_list_p.add_argument(
        "--tags",
        action="append",
        default=[],
        help="Filter by scenario tags (comma-separated or repeatable)",
    )
    mc_list_p.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format",
    )
    mc_list_p.add_argument(
        "--registry",
        help="Override the scenario registry path",
    )
    mc_validate_p = mc_sub.add_parser("validate", help="Validate Monte Carlo scenarios")
    mc_validate_p.add_argument(
        "scenario",
        nargs="?",
        help="Scenario name or config path (defaults to all registered scenarios)",
    )
    mc_validate_p.add_argument(
        "--tags",
        action="append",
        default=[],
        help="Filter by scenario tags (comma-separated or repeatable)",
    )
    mc_validate_p.add_argument(
        "--registry",
        help="Override the scenario registry path",
    )

    mc_run_p = mc_sub.add_parser("run", help="Run Monte Carlo scenarios")
    mc_run_p.add_argument("--scenario", required=True, help="Scenario name or config path")
    mc_run_p.add_argument(
        "--data",
        help="CSV/Parquet path for price or returns history (overrides base config)",
    )
    mc_run_p.add_argument("--out", help="Output directory for the Monte Carlo bundle")
    mc_run_p.add_argument(
        "--formats",
        action="append",
        default=[],
        help="Output formats (csv, json, parquet). Repeatable or comma-separated.",
    )
    mc_run_p.add_argument("--n-paths", type=int, help="Override number of Monte Carlo paths")
    mc_run_p.add_argument("--jobs", type=int, help="Override parallel job count")
    mc_run_p.add_argument("--seed", type=int, help="Override Monte Carlo seed")
    mc_run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate scenario configuration without executing",
    )
    mc_run_p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar output",
    )
    mc_run_p.add_argument(
        "--registry",
        help="Override the scenario registry path",
    )
    mc_viz_p = mc_sub.add_parser("viz", help="Render Monte Carlo chart artifacts from a bundle")
    mc_viz_p.add_argument(
        "--bundle",
        required=True,
        help="Path to Monte Carlo bundle directory containing summary/results files",
    )
    mc_viz_p.add_argument(
        "--out",
        required=True,
        help="Output directory for generated chart artifacts",
    )
    mc_viz_p.add_argument(
        "--charts",
        default="fan,path_dist,risk_return",
        help="Comma-separated chart identifiers (fan,path_dist,risk_return)",
    )
    mc_viz_p.add_argument(
        "--html",
        action="store_true",
        help="Export chart artifacts as HTML",
    )
    mc_viz_p.add_argument(
        "--json",
        action="store_true",
        help="Export chart artifacts as JSON",
    )
    mc_viz_p.add_argument(
        "--png",
        action="store_true",
        help="Export chart artifacts as PNG",
    )

    # Handle --check flag before parsing subcommands
    # This allows --check to work without requiring a subcommand
    if argv is None:
        argv = sys.argv[1:]

    if "--check" in argv:
        # Parse just to get the check flag, ignore subcommand requirement
        temp_parser = argparse.ArgumentParser(prog="trend-model", add_help=False)
        temp_parser.add_argument("--check", action="store_true")
        check_args, _ = temp_parser.parse_known_args(argv)
        if check_args.check:
            return check_environment()

    args = parser.parse_args(argv)

    log_suffix = getattr(args, "command", None) or "root"
    log_path = setup_logging(app_name=f"trend_cli_{log_suffix}")
    logging.getLogger(__name__).info("Log file initialised at %s", log_path)

    if args.check:
        return check_environment()

    if args.command == "gui":
        proc = subprocess.run(["streamlit", "run", str(APP_PATH)])
        return proc.returncode

    if args.command == "run":
        coverage_tracker: ConfigCoverageTracker | None = None
        if getattr(args, "config_coverage", False):
            coverage_tracker = ConfigCoverageTracker()
            activate_config_coverage(coverage_tracker)
        try:
            config_path = Path(args.config).resolve()
            if _should_handle_as_ui_config(config_path):
                return _run_from_ui_payload(
                    params_path=config_path,
                    data_path=Path(args.input),
                    auto_fix_dates=args.auto_fix_dates,
                    yes=args.yes,
                    no_cache=args.no_cache,
                    log_file=Path(args.log_file) if args.log_file else None,
                    structured_log=not args.no_structured_log,
                    bundle=Path(args.bundle) if args.bundle else None,
                )
            cfg = load_config(args.config)
            if coverage_tracker is not None:
                if not _maybe_track_config_coverage(config_path, args.input):
                    return 1
                wrap_config_for_coverage(cfg, coverage_tracker)
            if not _maybe_validate_config(
                cfg, base_path=config_path.parent, config_path=config_path
            ):
                return 1
            if args.preset:
                try:
                    spec_preset = get_trend_spec_preset(args.preset)
                except KeyError:
                    available = ", ".join(list_trend_spec_presets())
                    print(
                        f"Unknown preset '{args.preset}'. Available presets: {available}",
                        file=sys.stderr,
                    )
                    return 2
                _apply_trend_spec_preset(cfg, spec_preset)
            set_cache_enabled(not args.no_cache)
            if getattr(args, "preset", None):
                try:
                    portfolio_preset = get_trend_preset(args.preset)
                except KeyError:
                    available = ", ".join(list_preset_slugs())
                    print(
                        f"Unknown preset '{args.preset}'. Available: {available}",
                        file=sys.stderr,
                    )
                    return 2
                apply_trend_preset(cfg, portfolio_preset)
            cli_seed = args.seed
            env_seed = os.getenv("TREND_SEED")
            # Precedence: CLI flag > TREND_SEED > config.seed > default 42
            if cli_seed is not None:
                setattr(cfg, "seed", int(cli_seed))
            elif env_seed is not None and env_seed.isdigit():
                setattr(cfg, "seed", int(env_seed))
            data_section = getattr(cfg, "data", None)
            missing_policy = None
            missing_limit = None
            if isinstance(data_section, Mapping):
                missing_policy = data_section.get("missing_policy")
                missing_limit = data_section.get("missing_limit")
            else:
                missing_policy = getattr(data_section, "missing_policy", None)
                missing_limit = getattr(data_section, "missing_limit", None)

            if args.auto_fix_dates:
                try:
                    issues = inspect_ui_date_issues(args.input)
                except MarketDataValidationError as exc:
                    print(exc.user_message, file=sys.stderr)
                    for issue in exc.issues:
                        print(f"- {issue}", file=sys.stderr)
                    return 1
                has_fixable = issues.has_corrections or issues.total_droppable_rows > 0
                if has_fixable and not args.yes:
                    if not sys.stdin.isatty():
                        print(
                            "Date corrections require confirmation. Re-run with --yes to approve.",
                            file=sys.stderr,
                        )
                        return 1
                    prompt = (
                        f"Apply {len(issues.corrections)} date correction(s) and "
                        f"drop {issues.total_droppable_rows} row(s)? [y/N]: "
                    )
                    response = input(prompt).strip().lower()
                    if response not in {"y", "yes"}:
                        print("Cancelled date corrections.")
                        return 1

            try:
                loaded_frame, _, summary = load_ui_dataset(
                    args.input,
                    auto_fix_dates=args.auto_fix_dates,
                    missing_policy=missing_policy or "drop",
                    missing_limit=missing_limit,
                )
            except MarketDataValidationError as exc:
                print(exc.user_message, file=sys.stderr)
                for issue in exc.issues:
                    print(f"- {issue}", file=sys.stderr)
                return 1
            if summary.corrected_dates or summary.dropped_rows:
                parts: list[str] = []
                if summary.corrected_dates:
                    parts.append(f"{summary.corrected_dates} date correction(s)")
                if summary.dropped_rows:
                    parts.append(f"{summary.dropped_rows} row(s) dropped")
                print(f"Applied UI-style date fixes: {', '.join(parts)}")

            df = loaded_frame.reset_index()
            universe_spec: NamedUniverse | None = None
            if getattr(args, "universe", None):
                mask, universe_spec = load_universe(args.universe, prices=df)
                df = _apply_universe_mask(df, mask, date_column=universe_spec.date_column)
            if universe_spec is not None:
                _attach_universe_paths(cfg, universe_spec, csv_path=args.input)
            return _execute_analysis_run(
                cfg,
                df,
                config_path=config_path,
                input_path=Path(args.input),
                log_file=Path(args.log_file) if args.log_file else None,
                structured_log=not args.no_structured_log,
                bundle=Path(args.bundle) if args.bundle else None,
            )
        finally:
            if coverage_tracker is not None:
                print(coverage_tracker.format_report())
                deactivate_config_coverage()

    if args.command == "run-ui":
        print(
            "WARNING: 'trend-model run-ui' is deprecated. "
            "Use 'trend-model run' with the same --params JSON file instead.",
            file=sys.stderr,
        )
        return _run_from_ui_payload(
            params_path=Path(args.params),
            data_path=Path(args.data),
            auto_fix_dates=args.auto_fix_dates,
            yes=args.yes,
            no_cache=args.no_cache,
            log_file=Path(args.log_file) if args.log_file else None,
            structured_log=not args.no_structured_log,
            bundle=Path(args.bundle) if args.bundle else None,
        )

    if args.command == "mc":
        return _handle_mc_command(args)

    # This shouldn't be reached with required=True.
    return 0


def _parse_mc_tags(raw_tags: Sequence[str] | None) -> list[str]:
    if not raw_tags:
        return []
    tags: list[str] = []
    for raw in raw_tags:
        for tag in str(raw).split(","):
            cleaned = tag.strip()
            if cleaned:
                tags.append(cleaned)
    return tags


def _format_mc_tag_label(tags: Sequence[str]) -> str:
    if not tags:
        return "-"
    return ", ".join(sorted(dict.fromkeys(tag.strip() for tag in tags if tag.strip())))


def _render_mc_table(entries: Sequence[ScenarioRegistryEntry]) -> str:
    if not entries:
        return "No Monte Carlo scenarios found."

    rows = [
        {
            "Name": entry.name,
            "Tags": _format_mc_tag_label(entry.tags),
            "Description": entry.description or "",
            "Path": str(entry.path),
        }
        for entry in entries
    ]
    columns = ["Name", "Tags", "Description", "Path"]
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))

    header = "  ".join(col.ljust(widths[col]) for col in columns)
    divider = "  ".join("-" * widths[col] for col in columns)
    lines = [header, divider]
    for row in rows:
        lines.append("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))
    return "\n".join(lines)


def _resolve_mc_registry_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _load_mc_scenario_value(raw: str, *, registry_path: Path | None) -> MonteCarloScenario:
    if not raw:
        raise ValueError("Scenario name is required")
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return load_scenario_from_path(candidate)
    if candidate.suffix.lower() in {".yml", ".yaml"}:
        raise FileNotFoundError(f"Scenario config '{candidate}' does not exist")
    return load_scenario(raw, registry_path=registry_path)


def _parse_mc_formats(raw_formats: Sequence[str] | None) -> list[str]:
    if not raw_formats:
        return []
    if isinstance(raw_formats, str):
        items = [raw_formats]
    else:
        items = list(raw_formats)
    formats: list[str] = []
    for raw in items:
        for chunk in str(raw).split(","):
            cleaned = chunk.strip().lower()
            if cleaned:
                formats.append(cleaned)
    return formats


MC_OUTPUT_FORMATS = {"csv", "json", "parquet"}


def _validate_mc_formats(
    formats: Sequence[str] | str | None,
    *,
    label: str = "outputs.formats",
) -> list[str]:
    if formats is None:
        return []
    if isinstance(formats, str):
        raw_list = _parse_mc_formats([formats])
    else:
        raw_list = _parse_mc_formats([str(item) for item in formats])
    if not raw_list:
        return []
    invalid = sorted({fmt for fmt in raw_list if fmt not in MC_OUTPUT_FORMATS})
    if not invalid:
        return []
    return [f"{label} contains unsupported values: {', '.join(invalid)}"]


def _render_mc_output_dir(
    template: str,
    *,
    scenario_name: str,
    timestamp: str,
) -> Path:
    rendered = template.format(scenario_name=scenario_name, timestamp=timestamp)
    return Path(rendered)


def _require_mc_name(scenario: MonteCarloScenario) -> str:
    name = scenario.name
    if not name:
        raise ValueError("Monte Carlo scenario is missing a name")
    return name


def _require_mc_base_config(scenario: MonteCarloScenario) -> Path:
    base_config = scenario.base_config
    if base_config is None:
        raise ValueError("Monte Carlo scenario is missing base_config")
    if isinstance(base_config, Path):
        return base_config
    return Path(str(base_config))


def _require_mc_settings(scenario: MonteCarloScenario) -> MonteCarloSettings:
    settings = scenario.monte_carlo
    if not isinstance(settings, MonteCarloSettings):
        raise ValueError("monte_carlo settings are not resolved")
    return settings


def _resolve_mc_output_dir(
    scenario: MonteCarloScenario,
    *,
    override: str | None,
    timestamp: str,
) -> Path:
    if override:
        return Path(override)
    outputs = scenario.outputs
    if isinstance(outputs, Mapping):
        directory = outputs.get("directory")
        if directory:
            return _render_mc_output_dir(
                str(directory),
                scenario_name=_require_mc_name(scenario),
                timestamp=timestamp,
            )
    fallback = f"outputs/monte_carlo/{_require_mc_name(scenario)}/{timestamp}"
    return Path(fallback)


def _load_mc_price_history(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        validated = load_mc_market_data_parquet(str(path))
    else:
        validated = load_mc_market_data_csv(str(path))
    frame = validated.frame.copy()
    if validated.metadata.mode == MarketDataMode.RETURNS:
        if frame.empty:
            raise ValueError("returns data must not be empty")
        if (frame <= -1.0).any().any():
            raise ValueError("returns contain values <= -1; cannot convert to prices")
        return (1.0 + frame).cumprod() * 100.0
    return frame


def _validate_mc_scenario(scenario: MonteCarloScenario) -> list[str]:
    errors: list[str] = []
    runner: MonteCarloRunner | None = None
    try:
        runner = MonteCarloRunner(scenario)
    except Exception as exc:
        errors.append(f"base_config: {exc}")
        return errors

    base_config = runner.base_config
    base_path = _require_mc_base_config(scenario).parent

    return_model = scenario.return_model
    if isinstance(return_model, Mapping):
        kind = str(return_model.get("kind") or "stationary_bootstrap").lower()
        allowed = {
            "stationary_bootstrap",
            "bootstrap",
            "regime_bootstrap",
            "regime_conditioned",
        }
        if kind not in allowed:
            errors.append(f"return_model.kind must be one of: {', '.join(sorted(allowed))}")

    outputs = scenario.outputs
    if isinstance(outputs, Mapping):
        errors.extend(_validate_mc_formats(outputs.get("formats", outputs.get("format"))))

    try:
        strategies = runner.resolve_strategies()
    except Exception as exc:
        errors.append(f"strategy_set: {exc}")
        return errors

    for variant in strategies:
        try:
            variant.to_trend_config(base_config, base_path=base_path)
        except ValueError as exc:
            errors.append(str(exc))

    return errors


def _apply_mc_overrides(
    scenario: MonteCarloScenario,
    *,
    n_paths: int | None,
    jobs: int | None,
    seed: int | None,
) -> MonteCarloScenario:
    settings = _require_mc_settings(scenario)
    scenario.monte_carlo = MonteCarloSettings(
        mode=settings.mode,
        n_paths=n_paths if n_paths is not None else settings.n_paths,
        horizon_years=settings.horizon_years,
        frequency=settings.frequency,
        seed=seed if seed is not None else settings.seed,
        jobs=jobs if jobs is not None else settings.jobs,
    )
    return scenario


def _write_mc_manifest(
    output_dir: Path,
    *,
    scenario: MonteCarloScenario,
    results: MonteCarloResults,
    overrides: Mapping[str, Any],
    exported_files: Mapping[str, Path],
    data_path: Path | None,
    jobs_used: int,
) -> Path:
    settings = _require_mc_settings(scenario)
    scenario_name = _require_mc_name(scenario)
    payload = {
        "scenario": scenario_name,
        "description": scenario.description,
        "version": scenario.version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_config": str(scenario.base_config),
        "data_path": str(data_path) if data_path else None,
        "settings": {
            "mode": getattr(settings, "mode", None),
            "n_paths": getattr(settings, "n_paths", None),
            "horizon_years": getattr(settings, "horizon_years", None),
            "frequency": getattr(settings, "frequency", None),
            "seed": getattr(settings, "seed", None),
            "jobs": jobs_used,
        },
        "overrides": dict(overrides),
        "results": {
            "rows": int(results.results_frame.shape[0]),
            "summary_rows": int(results.summary_frame.shape[0]),
            "errors": len(results.errors),
        },
        "outputs": {
            "directory": str(output_dir),
            "files": {key: str(path) for key, path in exported_files.items()},
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _is_valid_tqdm_instance(candidate: Any) -> bool:
    if candidate is None:
        return False
    for attr in ("update", "refresh", "close"):
        if not callable(getattr(candidate, attr, None)):
            return False
    if not hasattr(candidate, "total"):
        return False
    total = getattr(candidate, "total")
    if total is None:
        return True
    if isinstance(total, numbers.Real):
        try:
            return float(total) >= 0
        except (TypeError, ValueError):
            return False
    return False


def _configure_tqdm_instance(
    candidate: Any,
    *,
    total: int,
) -> tuple[Any, bool]:
    expected: dict[str, Any] = {"total": total, "unit": "path", "file": sys.stderr}
    mismatches: dict[str, Any] = {}

    for attr, expected_value in expected.items():
        if not hasattr(candidate, attr):
            continue
        current_value = getattr(candidate, attr)
        if current_value != expected_value:
            mismatches[attr] = expected_value

    if not mismatches:
        return candidate, True

    for attr, expected_value in mismatches.items():
        try:
            setattr(candidate, attr, expected_value)
        except Exception:
            continue

    remaining = []
    for attr, expected_value in mismatches.items():
        if not hasattr(candidate, attr):
            continue
        if getattr(candidate, attr) != expected_value:
            remaining.append(attr)

    if not remaining:
        return candidate, True

    try:
        replacement = type(candidate)(total=total, unit="path", file=sys.stderr)
    except Exception:
        logging.getLogger(__name__).warning(
            "Provided tqdm instance could not be reconfigured; falling back to text progress."
        )
        return candidate, False

    if _is_valid_tqdm_instance(replacement):
        return replacement, True

    logging.getLogger(__name__).warning(
        "Provided tqdm instance could not be reconfigured; falling back to text progress."
    )
    return candidate, False


def _build_mc_progress_callback(
    *,
    total: int,
    enabled: bool,
) -> tuple[Callable[[Mapping[str, Any]], None] | None, Callable[[], None]]:
    if not enabled:
        return None, lambda: None

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    if tqdm is None:
        state = {"last": -1}

        def _text_callback(payload: Mapping[str, Any]) -> None:
            completed = int(payload.get("completed", 0))
            total_value = int(payload.get("total", total))
            if completed == state["last"]:
                return
            state["last"] = completed
            print(f"Progress: {completed}/{total_value}", file=sys.stderr)

        return _text_callback, lambda: None

    bar = None
    if _is_valid_tqdm_instance(tqdm):
        bar, configured = _configure_tqdm_instance(tqdm, total=total)
        if not configured:
            bar = None
    elif callable(tqdm):
        try:
            bar = tqdm(total=total, unit="path", file=sys.stderr)
        except Exception:
            bar = None
    if bar is None or not _is_valid_tqdm_instance(bar):
        state = {"last": -1}

        def _text_callback(payload: Mapping[str, Any]) -> None:
            completed = int(payload.get("completed", 0))
            total_value = int(payload.get("total", total))
            if completed == state["last"]:
                return
            state["last"] = completed
            print(f"Progress: {completed}/{total_value}", file=sys.stderr)

        return _text_callback, lambda: None

    state = {"completed": 0}

    def _callback(payload: Mapping[str, Any]) -> None:
        completed = int(payload.get("completed", 0))
        total_value = int(payload.get("total", total))
        if bar.total != total_value:
            bar.total = total_value
        delta = completed - state["completed"]
        if delta > 0:
            bar.update(delta)
        else:
            bar.refresh()
        state["completed"] = completed

    def _close() -> None:
        bar.close()

    return _callback, _close


def _validate_mc_viz_output_flags(args: argparse.Namespace) -> None:
    if not any(
        (getattr(args, "html", False), getattr(args, "json", False), getattr(args, "png", False))
    ):
        raise ValueError(
            "The 'mc viz' command requires at least one output flag: --html, --json, or --png"
        )


def _read_mc_frame(path: Path, *, label: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in {".parquet", ".csv", ".json"}:
        raise ValueError(f"Unsupported {label} file format '{path.suffix}' for '{path.name}'.")
    try:
        if suffix == ".parquet":
            frame = pd.read_parquet(path)
        elif suffix == ".csv":
            frame = pd.read_csv(path)
        else:
            frame = pd.read_json(path)
    except Exception as exc:
        raise ValueError(f"Failed to read {label} data from '{path}': {exc}") from exc
    if isinstance(frame, pd.Series):
        return frame.to_frame()
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"Expected {label} data in '{path}' to load as a table.")
    return frame


def _load_mc_frame(bundle_dir: Path, *, stem: str) -> pd.DataFrame:
    candidates = tuple(bundle_dir / f"{stem}.{ext}" for ext in ("parquet", "csv", "json"))
    existing = next((candidate for candidate in candidates if candidate.exists()), None)
    if existing is None:
        expected = ", ".join(path.name for path in candidates)
        raise FileNotFoundError(
            f"Missing required MC {stem} file in '{bundle_dir}'. Expected one of: {expected}"
        )
    return _read_mc_frame(existing, label=stem)


def _load_mc_summary_frame(bundle_dir: Path) -> pd.DataFrame:
    return _load_mc_frame(bundle_dir, stem="summary")


def _load_mc_results_frame(bundle_dir: Path) -> pd.DataFrame:
    return _load_mc_frame(bundle_dir, stem="results")


def _load_mc_bundle_frames(bundle: str | os.PathLike[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle_dir = Path(bundle).expanduser().resolve()
    if not bundle_dir.exists():
        raise FileNotFoundError(f"MC bundle directory does not exist: {bundle_dir}")
    if not bundle_dir.is_dir():
        raise NotADirectoryError(f"MC bundle path is not a directory: {bundle_dir}")
    required_stems = ("summary", "results")
    missing_inputs: list[str] = []
    expected_by_stem: dict[str, str] = {}
    for stem in required_stems:
        candidates = tuple(bundle_dir / f"{stem}.{ext}" for ext in ("parquet", "csv", "json"))
        if not any(candidate.exists() for candidate in candidates):
            missing_inputs.append(stem)
            expected_by_stem[stem] = ", ".join(path.name for path in candidates)
    if missing_inputs:
        if len(missing_inputs) == 1:
            stem = missing_inputs[0]
            expected = expected_by_stem[stem]
            raise FileNotFoundError(
                f"Missing required MC {stem} file in '{bundle_dir}'. Expected one of: {expected}"
            )
        missing_text = ", ".join(missing_inputs)
        expected_text = "; ".join(f"{stem}: {expected_by_stem[stem]}" for stem in missing_inputs)
        raise FileNotFoundError(
            f"Missing required MC input files in '{bundle_dir}': {missing_text}. "
            f"Expected one of each: {expected_text}"
        )
    return _load_mc_summary_frame(bundle_dir), _load_mc_results_frame(bundle_dir)


def _load_mc_nav_paths_frame(bundle: str | os.PathLike[str]) -> pd.DataFrame | None:
    bundle_dir = Path(bundle).expanduser().resolve()
    nav_paths_path = bundle_dir / "nav_paths.parquet"
    if not nav_paths_path.exists():
        return None
    return _read_mc_frame(nav_paths_path, label="nav_paths")


def _parse_mc_chart_selection(charts_value: str) -> list[str]:
    requested = [token.strip().lower() for token in charts_value.split(",") if token.strip()]
    if not requested:
        raise ValueError("The 'mc viz' command requires at least one chart in --charts.")

    seen: set[str] = set()
    ordered: list[str] = []
    for chart in requested:
        if chart not in seen:
            seen.add(chart)
            ordered.append(chart)

    supported = tuple(_mc_chart_builders().keys())
    unsupported = [chart for chart in ordered if chart not in supported]
    if unsupported:
        supported_text = ", ".join(supported)
        invalid_text = ", ".join(unsupported)
        raise ValueError(
            f"Unsupported chart identifier(s): {invalid_text}. Supported charts: {supported_text}"
        )
    return ordered


def _mc_nav_source_frame(
    summary_frame: pd.DataFrame,
    results_frame: pd.DataFrame,
    nav_paths_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    if nav_paths_frame is not None:
        return nav_paths_frame

    for frame in (results_frame, summary_frame):
        numeric = frame.select_dtypes(include=[np.number]).copy()
        numeric = numeric.dropna(how="all")
        if not numeric.empty:
            return numeric

    raise ValueError(
        "Unable to derive path data for Monte Carlo charts. "
        "Provide nav_paths.parquet or numeric summary/results files."
    )


def _build_mc_fan_chart(
    summary_frame: pd.DataFrame,
    results_frame: pd.DataFrame,
    nav_paths_frame: pd.DataFrame | None,
) -> Any:
    from trend_analysis.viz import fan

    nav_frame = _mc_nav_source_frame(summary_frame, results_frame, nav_paths_frame)
    return fan.make(nav_frame)


def _build_mc_path_dist_chart(
    summary_frame: pd.DataFrame,
    results_frame: pd.DataFrame,
    nav_paths_frame: pd.DataFrame | None,
) -> Any:
    from trend_analysis.viz import path_dist

    nav_frame = _mc_nav_source_frame(summary_frame, results_frame, nav_paths_frame)
    return path_dist.make(nav_frame)


def _build_mc_risk_return_chart(
    summary_frame: pd.DataFrame,
    results_frame: pd.DataFrame,
    nav_paths_frame: pd.DataFrame | None,
) -> Any:
    from trend_analysis.viz import risk_return

    nav_frame = _mc_nav_source_frame(summary_frame, results_frame, nav_paths_frame)
    returns_frame = nav_frame.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    returns_frame = returns_frame.dropna(how="all")
    if returns_frame.empty:
        returns_frame = nav_frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    return risk_return.make(returns_frame)


def _mc_chart_builders() -> (
    dict[str, Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame | None], Any]]
):
    return {
        "fan": _build_mc_fan_chart,
        "path_dist": _build_mc_path_dist_chart,
        "risk_return": _build_mc_risk_return_chart,
    }


def _export_mc_chart_artifacts(
    charts: Mapping[str, Any],
    out_dir: Path,
    *,
    include_html: bool,
    include_json: bool,
    include_png: bool,
) -> tuple[Path, list[str]]:
    from trend_analysis.monte_carlo.export_bundle import save as export_bundle

    plots_dir = out_dir.expanduser().resolve() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = plots_dir / "mc_charts_bundle.zip"

    warnings: list[str] = []
    export_bundle(
        charts,
        destination=bundle_path,
        include_html=include_html,
        include_json=include_json,
        include_png=include_png,
        warnings=warnings,
    )
    with zipfile.ZipFile(bundle_path) as archive:
        archive.extractall(plots_dir)
    return plots_dir, warnings


def _run_mc_viz_command(args: argparse.Namespace) -> int:
    from trend.mc.viz import execute_mc_viz

    return execute_mc_viz(
        bundle_path=args.bundle,
        out_dir=args.out,
        charts=args.charts,
        html=args.html,
        json=args.json,
        png=args.png,
    )


def _handle_mc_command(args: argparse.Namespace) -> int:
    """Dispatch Monte Carlo CLI commands."""
    subcommand = getattr(args, "mc_command", None)
    if subcommand == "list":
        tags = _parse_mc_tags(getattr(args, "tags", None))
        registry_path = _resolve_mc_registry_path(getattr(args, "registry", None))
        try:
            registry_entries = list_scenarios(tags=tags, registry_path=registry_path)
        except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
            print(f"Failed to list Monte Carlo scenarios: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Failed to list Monte Carlo scenarios: {exc}", file=sys.stderr)
            return 2
        output_format = getattr(args, "format", "table")
        if output_format == "json":
            payload = [
                {
                    "name": entry.name,
                    "description": entry.description,
                    "tags": list(entry.tags),
                    "path": str(entry.path),
                }
                for entry in registry_entries
            ]
            print(json.dumps(payload, indent=2))
        else:
            print(_render_mc_table(registry_entries))
        return 0
    if subcommand == "validate":
        registry_path = _resolve_mc_registry_path(getattr(args, "registry", None))
        tags = _parse_mc_tags(getattr(args, "tags", None))
        scenario_arg = getattr(args, "scenario", None)
        scenarios: list[MonteCarloScenario] = []
        if scenario_arg:
            try:
                scenarios = [_load_mc_scenario_value(scenario_arg, registry_path=registry_path)]
            except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
                print(f"Scenario validation failed: {exc}", file=sys.stderr)
                return 1
            except Exception as exc:
                print(f"Scenario validation failed: {exc}", file=sys.stderr)
                return 2
        else:
            try:
                entries = list_scenarios(tags=tags, registry_path=registry_path)
            except (ValueError, FileNotFoundError) as exc:
                print(f"Scenario registry error: {exc}", file=sys.stderr)
                return 1
            except Exception as exc:
                print(f"Scenario registry error: {exc}", file=sys.stderr)
                return 2
            for entry in entries:
                try:
                    scenarios.append(load_scenario(entry.name, registry_path=registry_path))
                except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
                    print(
                        f"Scenario '{entry.name}' failed to load: {exc}",
                        file=sys.stderr,
                    )
                    return 1
                except Exception as exc:
                    print(
                        f"Scenario '{entry.name}' failed to load: {exc}",
                        file=sys.stderr,
                    )
                    return 2

        failures = 0
        for scenario in scenarios:
            errors = _validate_mc_scenario(scenario)
            if errors:
                failures += 1
                print(f"Scenario '{scenario.name}' failed validation:", file=sys.stderr)
                for error in errors:
                    print(f"- {error}", file=sys.stderr)
            else:
                print(f"Scenario '{scenario.name}': OK")
        return 0 if failures == 0 else 1
    if subcommand == "run":
        registry_path = _resolve_mc_registry_path(getattr(args, "registry", None))
        scenario_arg = getattr(args, "scenario", None) or ""
        try:
            scenario = _load_mc_scenario_value(scenario_arg, registry_path=registry_path)
        except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
            print(f"Scenario run failed: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Scenario run failed: {exc}", file=sys.stderr)
            return 2

        overrides = {
            key: value
            for key, value in {
                "n_paths": getattr(args, "n_paths", None),
                "jobs": getattr(args, "jobs", None),
                "seed": getattr(args, "seed", None),
            }.items()
            if value is not None
        }
        try:
            _apply_mc_overrides(
                scenario,
                n_paths=getattr(args, "n_paths", None),
                jobs=getattr(args, "jobs", None),
                seed=getattr(args, "seed", None),
            )
        except ValueError as exc:
            print(f"Scenario run failed: {exc}", file=sys.stderr)
            return 1

        format_overrides = _parse_mc_formats(getattr(args, "formats", None))
        validation_errors = _validate_mc_scenario(scenario)
        validation_errors.extend(
            _validate_mc_formats(format_overrides, label="format overrides")
            if format_overrides
            else []
        )
        if validation_errors:
            print(f"Scenario '{scenario.name}' failed validation:", file=sys.stderr)
            for error in validation_errors:
                print(f"- {error}", file=sys.stderr)
            return 1

        if getattr(args, "dry_run", False):
            print(f"Scenario '{scenario.name}' validated. Dry run complete.")
            return 0

        data_path = getattr(args, "data", None)
        price_history = None
        if data_path:
            try:
                price_history = _load_mc_price_history(Path(data_path))
            except (MarketDataValidationError, ValueError) as exc:
                print(f"Scenario run failed: {exc}", file=sys.stderr)
                return 1

        settings = _require_mc_settings(scenario)
        total_paths = int(settings.n_paths) if settings.n_paths is not None else 0
        progress_enabled = not getattr(args, "no_progress", False)
        progress_cb, progress_close = _build_mc_progress_callback(
            total=total_paths,
            enabled=progress_enabled,
        )

        try:
            runner = MonteCarloRunner(
                scenario,
                base_config=None,
                price_history=price_history,
            )
            results = runner.run(
                progress_callback=progress_cb,
                jobs=getattr(args, "jobs", None),
            )
        except Exception as exc:
            print(f"Scenario run failed: {exc}", file=sys.stderr)
            return 2
        finally:
            progress_close()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = _resolve_mc_output_dir(
            scenario,
            override=getattr(args, "out", None),
            timestamp=timestamp,
        )
        outputs = scenario.outputs if isinstance(scenario.outputs, Mapping) else {}
        output_formats = format_overrides or _parse_mc_formats(
            outputs.get("formats", outputs.get("format", [])) if outputs else []
        )
        if not output_formats:
            output_formats = ["csv"]

        try:
            exported = export_results(results, output_dir, formats=output_formats)
        except Exception as exc:
            print(f"Scenario run failed: {exc}", file=sys.stderr)
            return 2

        jobs_used = runner._resolve_jobs(getattr(args, "jobs", None))
        _write_mc_manifest(
            output_dir,
            scenario=scenario,
            results=results,
            overrides=overrides,
            exported_files=exported,
            data_path=Path(data_path) if data_path else None,
            jobs_used=jobs_used,
        )

        if results.errors:
            print(
                f"Monte Carlo run completed with {len(results.errors)} error(s).",
                file=sys.stderr,
            )
        else:
            print(f"Monte Carlo run completed. Output: {output_dir}")
        return 0
    if subcommand == "viz":
        try:
            return _run_mc_viz_command(args)
        except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
            print(f"Monte Carlo viz failed: {exc}", file=sys.stderr)
            return 1
        except RuntimeError as exc:
            # Catches TrendCLIError (a RuntimeError subclass) raised by the
            # shared mc viz module for user-facing validation errors.
            print(f"Monte Carlo viz failed: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Monte Carlo viz failed: {exc}", file=sys.stderr)
            return 2
    print("Unknown Monte Carlo command.", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Unified CLI compatibility layer
# ---------------------------------------------------------------------------


def _load_configuration(path: str) -> tuple[Path, Any]:
    """Delegate to the unified CLI loader for backwards-compatibility."""

    from trend.cli import _load_configuration as unified_load_configuration

    result = unified_load_configuration(path)
    return cast(tuple[Path, Any], result)


def _resolve_returns_path(config_path: Path, cfg: Any, override: str | None) -> Path:
    """Reuse the unified CLI's returns resolution helper."""

    from trend.cli import _resolve_returns_path as unified_resolve_returns_path

    return unified_resolve_returns_path(config_path, cfg, override)


def _ensure_dataframe(path: Path) -> pd.DataFrame:
    """Proxy to the unified CLI dataframe loader."""

    from trend.cli import _ensure_dataframe as unified_ensure_dataframe

    return unified_ensure_dataframe(path)


def _run_pipeline(
    cfg: Any,
    returns_df: pd.DataFrame,
    *,
    source_path: Path | None,
    log_file: Path | None,
    structured_log: bool,
    bundle: Path | None,
) -> tuple[Any, str, Path | None]:
    """Call the unified CLI pipeline execution helper."""

    from trend.cli import _run_pipeline as unified_run_pipeline

    return unified_run_pipeline(
        cfg,
        returns_df,
        source_path=source_path,
        log_file=log_file,
        structured_log=structured_log,
        bundle=bundle,
    )


def _print_summary(cfg: Any, result: Any) -> None:
    """Defer to the shared summary printer used by ``trend.cli``."""

    from trend.cli import _print_summary as unified_print_summary

    return unified_print_summary(cfg, result)


def _write_report_files(out_dir: Path, cfg: Any, result: Any, *, run_id: str) -> None:
    """Forward report artefact writes to the unified CLI implementation."""

    from trend.cli import _write_report_files as unified_write_report_files

    return unified_write_report_files(out_dir, cfg, result, run_id=run_id)
