"""Report, export, and pipeline-output implementations for ``trend.cli``."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import pandas as pd

from trend.cli_support import (
    extract_cache_stats,
    maybe_log_step,
)
from trend.diagnostics import DiagnosticPayload, DiagnosticResult
from trend.mc.viz import TrendCLIError
from trend_analysis import export
from trend_analysis import logging as run_logging
from trend_analysis.api import RunResult, run_simulation
from trend_analysis.config.schema_validation import load_config as load_config_yaml
from trend_analysis.constants import DEFAULT_OUTPUT_DIRECTORY, DEFAULT_OUTPUT_FORMATS
from trend_analysis.export.run_envelope import write_run_envelope
from trend_analysis.identity import IdentityMap
from trend_analysis.logging_setup import setup_logging
from trend_analysis.reporting.portfolio_series import select_primary_portfolio_series
from trend_analysis.reporting.run_artifacts import write_run_artifacts
from trend_analysis.util.hash import working_run_id
from trend_analysis.util.json_compat import (
    JSON_UNSUPPORTED,
    json_compatible,
    json_primitive,
)
from trend_analysis.util.paths import proj_path

logger = logging.getLogger(__name__)


class _PerfLoggerState:
    last_path: Path | None = None
    diagnostic: DiagnosticPayload | None = None


_PERF_LOG_STATE = _PerfLoggerState()


def _report_pipeline_diagnostic(
    diagnostic: DiagnosticPayload,
    *,
    structured_log: bool,
    run_id: str,
) -> None:
    """Surface pipeline diagnostics within the public CLI."""

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


def _init_perf_logger(app_name: str = "app") -> DiagnosticResult[Path]:
    """Initialise central logging for CLI invocations.

    Returns the file path when logging is enabled, otherwise ``None``.
    """

    disable = os.environ.get("TREND_DISABLE_PERF_LOGS", "").strip().lower()
    if disable in {"1", "true", "yes"}:
        diagnostic = DiagnosticPayload(
            reason_code="PERF_LOG_DISABLED",
            message="Performance logging disabled via environment flag.",
        )
        _PERF_LOG_STATE.diagnostic = diagnostic
        return DiagnosticResult(value=None, diagnostic=diagnostic)
    try:
        log_path = setup_logging(app_name=app_name)
    except Exception as exc:  # pragma: no cover - fail-safe path
        logger.warning("Failed to initialise perf log handler: %s", exc)
        diagnostic = DiagnosticPayload(
            reason_code="PERF_LOG_DISABLED",
            message="Performance logging disabled or could not be initialised.",
            context={"error": str(exc)},
        )
        _PERF_LOG_STATE.diagnostic = diagnostic
        return DiagnosticResult(value=None, diagnostic=diagnostic)
    print(f"Run log: {log_path}")
    _PERF_LOG_STATE.last_path = log_path
    _PERF_LOG_STATE.diagnostic = None
    return DiagnosticResult.success(log_path)


def get_last_perf_log_path() -> Path | None:
    """Return the most recent CLI perf log path, if any."""

    return _PERF_LOG_STATE.last_path


def _resolved_export_settings(cfg: Any) -> tuple[str, list[str] | str, str] | None:
    export_cfg = getattr(cfg, "export", {}) or {}
    out_dir = export_cfg.get("directory")
    out_formats = export_cfg.get("formats")
    filename = export_cfg.get("filename", "analysis")
    if not out_dir and not out_formats:
        out_dir = DEFAULT_OUTPUT_DIRECTORY
        out_formats = DEFAULT_OUTPUT_FORMATS
    if not out_dir or not out_formats:
        return None
    formats = cast(list[str] | str, out_formats)
    return str(out_dir), formats, str(filename)


def _summary_text(cfg: Any, details: Any) -> str:
    split = getattr(cfg, "sample_split", {})
    return cast(
        str,
        export.format_summary_text(
            details,
            str(split.get("in_start", "")),
            str(split.get("in_end", "")),
            str(split.get("out_start", "")),
            str(split.get("out_end", "")),
        ),
    )


def _prepare_export_config(cfg: Any, directory: Path | None, formats: Iterable[str] | None) -> None:
    if directory is None and formats is None:
        return
    export_cfg = dict(getattr(cfg, "export", {}) or {})
    if directory is not None:
        export_cfg["directory"] = str(directory)
    if formats is not None:
        export_cfg["formats"] = [f for f in formats]
    try:
        setattr(cfg, "export", export_cfg)
    except Exception as exc:
        logger.warning("Failed to apply export configuration overrides: %s", exc)


def _run_pipeline(
    cfg: Any,
    returns_df: pd.DataFrame,
    *,
    source_path: Path | None,
    log_file: Path | None,
    structured_log: bool,
    bundle: Path | None,
) -> tuple[RunResult, str, Path | None]:
    _require_transaction_cost_controls(cfg)
    perf_log_result = _init_perf_logger()
    if perf_log_result.diagnostic:
        logger.info(perf_log_result.diagnostic.message)
    run_id = working_run_id(cfg, source_path)
    try:
        setattr(cfg, "run_id", run_id)
    except Exception as exc:
        logger.warning("Failed to apply run_id to config: %s", exc)

    log_path = None
    if structured_log:
        log_path = log_file or run_logging.get_default_log_path(run_id)
        run_logging.init_run_logger(run_id, log_path)
    maybe_log_step(structured_log, run_id, "start", "trend CLI execution started")
    maybe_log_step(
        structured_log,
        run_id,
        "load_data",
        "Loaded returns dataframe",
        rows=len(returns_df),
    )

    result = run_simulation(cfg, returns_df)
    maybe_log_step(
        structured_log,
        run_id,
        "pipeline_complete",
        "Pipeline execution finished",
        metrics_rows=len(result.metrics),
    )
    diagnostic = getattr(result, "diagnostic", None)
    if diagnostic and not result.details:
        _report_pipeline_diagnostic(
            diagnostic,
            structured_log=structured_log,
            run_id=run_id,
        )
        return result, run_id, log_path
    analysis = getattr(result, "analysis", None)
    # The following attributes are already set by run_simulation when analysis exists,
    # but we need to backfill them when analysis is absent (legacy callers).
    details = result.details
    if isinstance(details, dict):
        if analysis is None:
            portfolio_series = select_primary_portfolio_series(details, prefer_raw=True)
            if portfolio_series is not None:
                setattr(result, "portfolio", portfolio_series)
        benchmarks = details.get("benchmarks")
        if isinstance(benchmarks, dict) and benchmarks:
            first = next(iter(benchmarks.values()))
            setattr(result, "benchmark", first)
        weights_user = details.get("weights_user_weight")
        if weights_user is not None:
            setattr(result, "weights", weights_user)

    maybe_log_step(
        structured_log,
        run_id,
        "summary_render",
        "Simulation finished and summary rendered",
    )

    _handle_exports(cfg, result, structured_log, run_id)
    ledger_result = _persist_turnover_ledger(run_id, getattr(result, "details", {}))
    if ledger_result.diagnostic:
        logger.info(ledger_result.diagnostic.message)

    if bundle:
        _write_bundle(cfg, result, source_path, Path(bundle), structured_log, run_id)

    return result, run_id, log_path


def _finish_structured_log(
    enabled: bool,
    run_id: str,
    log_path: Path | None,
    result: RunResult | None,
) -> None:
    """Record the terminal event after every run artifact has been written."""

    cache_stats = extract_cache_stats(result.details) if result is not None else {}
    if cache_stats:
        maybe_log_step(
            enabled,
            run_id,
            "cache_stats",
            "Cache statistics summary",
            **cache_stats,
        )
    maybe_log_step(
        enabled,
        run_id,
        "end",
        "CLI run complete",
        log_file=str(log_path) if log_path is not None else "",
    )


def _handle_exports(cfg: Any, result: RunResult, structured_log: bool, run_id: str) -> None:
    resolved = _resolved_export_settings(cfg)
    if resolved is None:
        return
    out_dir, out_formats, filename = resolved
    format_list = [out_formats] if isinstance(out_formats, str) else list(out_formats)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    maybe_log_step(
        structured_log,
        run_id,
        "export_start",
        "Beginning export",
        formats=format_list,
    )
    data = {"metrics": result.metrics}
    export.append_narrative_section(data, result.details, config=cfg)
    split = getattr(cfg, "sample_split", {})
    in_start = str(split.get("in_start")) if split else ""
    in_end = str(split.get("in_end")) if split else ""
    out_start = str(split.get("out_start")) if split else ""
    out_end = str(split.get("out_end")) if split else ""
    if any(fmt.lower() == "xlsx" for fmt in format_list):
        formatter = export.make_summary_formatter(
            result.details, in_start, in_end, out_start, out_end
        )
        data["summary"] = export.summary_frame_from_result(result.details)
        export.export_to_excel(
            data,
            str(out_dir_path / f"{filename}.xlsx"),
            default_sheet_formatter=formatter,
        )
        remaining = [fmt for fmt in format_list if fmt.lower() != "xlsx"]
        if remaining:
            export.export_data(
                data,
                str(out_dir_path / filename),
                formats=remaining,
            )
    else:
        export.export_data(
            data,
            str(out_dir_path / filename),
            formats=format_list,
        )
    maybe_log_step(structured_log, run_id, "export_complete", "Export done")


def _resolve_export_artifact_paths(
    out_dir: Path, filename: str, keys: Sequence[str], formats: Sequence[str]
) -> list[Path]:
    """Return the export artifact paths expected from ``export.export_data``."""

    seen: set[Path] = set()
    paths: list[Path] = []
    for fmt in formats:
        fmt_norm = (fmt or "").lower()
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


def _write_trend_run_artifacts(
    *,
    cfg: Any,
    result: RunResult,
    config_path: Path,
    input_path: Path,
    data_frame: pd.DataFrame,
    run_id: str,
    structured_log: bool,
) -> Path | None:
    """Write the replay manifest and run envelope used by ``trend run``."""

    resolved = _resolved_export_settings(cfg)
    if resolved is None:
        return None
    out_dir, out_formats, filename = resolved

    out_dir_path = Path(out_dir)
    fmt_list = [out_formats] if isinstance(out_formats, str) else list(out_formats)
    data_keys = ["metrics"]
    if isinstance(result.details, Mapping):
        narrative_data: dict[str, Any] = {"metrics": result.metrics}
        export.append_narrative_section(narrative_data, result.details, config=cfg)
        data_keys = list(narrative_data.keys())
        if any(fmt.lower() == "xlsx" for fmt in fmt_list):
            data_keys.append("summary")
    artifact_paths = _resolve_export_artifact_paths(out_dir_path, filename, data_keys, fmt_list)
    summary_text = _summary_text(cfg, result.details)
    try:
        raw_config_payload = load_config_yaml(config_path)
        config_payload: Any
        if hasattr(cfg, "model_dump"):
            config_payload = cfg.model_dump()
        elif hasattr(cfg, "__dict__"):
            config_payload = dict(getattr(cfg, "__dict__"))
        else:
            config_payload = cfg
        manifest_dir = write_run_artifacts(
            output_dir=out_dir_path,
            run_id=run_id,
            config=config_payload,
            config_path=str(config_path),
            input_path=input_path,
            data_frame=data_frame,
            metrics_frame=result.metrics,
            run_details=result.details if isinstance(result.details, Mapping) else {},
            exported_files=artifact_paths,
            summary_text=summary_text,
            identity_map=IdentityMap.from_config(
                raw_config_payload,
                base_path=config_path.parent,
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive parity with legacy CLI
        logger.warning("Failed to write run artifacts: %s", exc)
        return None
    maybe_log_step(
        structured_log,
        run_id,
        "run_artifacts",
        "Run manifest written",
        directory=str(manifest_dir),
    )
    try:
        envelope_path = write_run_envelope(
            result,
            config=config_payload,
            manifest_path=manifest_dir / "manifest.json",
            run_dir=manifest_dir,
        )
    except Exception as exc:  # pragma: no cover - defensive parity with manifest writer
        logger.warning("Failed to write run envelope: %s", exc)
    else:
        maybe_log_step(
            structured_log,
            run_id,
            "run_envelope",
            "Run envelope written",
            path=str(envelope_path),
        )
    return manifest_dir


def _bundle_config_payload(cfg: Any) -> dict[str, Any]:
    """Return JSON-serializable config metadata for reproducibility bundles."""

    from trend_analysis.util.hash import normalise_for_json

    if hasattr(cfg, "model_dump"):
        try:
            payload: Any = cfg.model_dump()
        except TypeError:  # pragma: no cover - defensive for exotic configs
            payload = dict(getattr(cfg, "__dict__", {}))
    elif hasattr(cfg, "__dict__"):
        payload = dict(getattr(cfg, "__dict__"))
    else:
        payload = cfg if isinstance(cfg, dict) else {}
    if isinstance(payload, dict):
        payload = {
            key: value
            for key, value in payload.items()
            if key not in {"_trend_run_spec", "trend_spec", "backtest_spec"}
        }
    normalised = normalise_for_json(payload)
    return normalised if isinstance(normalised, dict) else {"config": normalised}


def _write_bundle(
    cfg: Any,
    result: RunResult,
    source_path: Path | None,
    bundle_path: Path,
    structured_log: bool,
    run_id: str,
) -> None:
    from trend_analysis.export.bundle import export_bundle

    bundle_path = bundle_path.resolve()
    if bundle_path.is_dir():
        bundle_path = bundle_path / "analysis_bundle.zip"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    # Attach metadata expected by export_bundle
    setattr(result, "config", _bundle_config_payload(cfg))
    if source_path is not None:
        setattr(result, "input_path", source_path)
    export_bundle(result, bundle_path)
    print(f"Bundle written: {bundle_path}")
    maybe_log_step(
        structured_log,
        run_id,
        "bundle_complete",
        "Reproducibility bundle created",
        bundle=str(bundle_path),
    )


def _print_summary(cfg: Any, result: RunResult) -> None:
    text = _summary_text(cfg, result.details)
    print(text)
    cache_stats = extract_cache_stats(result.details)
    if cache_stats:
        print("\nCache statistics:")
        for key, value in cache_stats.items():
            print(f"  {key.capitalize()}: {value}")


def _write_report_files(out_dir: Path, cfg: Any, result: RunResult, *, run_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"metrics_{run_id}.csv"
    result.metrics.to_csv(metrics_path)
    summary_path = out_dir / f"summary_{run_id}.txt"
    summary_text = _summary_text(cfg, result.details)
    summary_path.write_text(summary_text, encoding="utf-8")
    details_path = out_dir / f"details_{run_id}.json"
    with details_path.open("w", encoding="utf-8") as fh:
        json.dump(result.details, fh, default=_json_default, indent=2)
    turnover_csv_result = _maybe_write_turnover_csv(out_dir, getattr(result, "details", {}))
    if turnover_csv_result.diagnostic:
        logger.info(turnover_csv_result.diagnostic.message)
    print(f"Report artefacts written to {out_dir}")


def _resolve_report_output_path(output: str | None, export_dir: Path | None, run_id: str) -> Path:
    if output:
        base = Path(output).expanduser()
        if base.exists() and base.is_dir():
            return base / f"trend_report_{run_id}.html"
        if base.suffix.lower() in {".html", ".htm"}:
            return base
        if base.suffix:
            return base
        return base / f"trend_report_{run_id}.html"
    base_dir = export_dir if export_dir is not None else proj_path()
    return base_dir / f"trend_report_{run_id}.html"


def _json_default(obj: Any) -> Any:  # pragma: no cover - helper
    if is_dataclass(obj) and not isinstance(obj, type):
        return json_compatible(asdict(obj))
    if isinstance(obj, pd.Series):
        data: dict[str | int | float, Any] = {}
        for key, value in obj.items():
            coerced_key: str | int | float
            if isinstance(key, (str, int, float)):
                coerced_key = key
            else:
                coerced_key = str(key)
            data[coerced_key] = json_compatible(value)
        return data
    if isinstance(obj, pd.DataFrame):
        result: dict[str | int | float, Any] = {}
        for col in obj.columns:
            coerced_col: str | int | float
            if isinstance(col, (str, int, float)):
                coerced_col = col
            else:
                coerced_col = str(col)
            result[coerced_col] = _json_default(obj[col])
        return result
    primitive = json_primitive(obj)
    if primitive is not JSON_UNSUPPORTED:
        return primitive
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _maybe_write_turnover_csv(directory: Path, details: Any) -> DiagnosticResult[Path]:
    if not isinstance(details, Mapping):
        return DiagnosticResult.failure(
            reason_code="NO_TURNOVER_EXPORT",
            message="Turnover diagnostics absent or non-numeric; skipping CSV export.",
            context={"details_type": type(details).__name__},
        )
    diag = details.get("risk_diagnostics")
    if not isinstance(diag, Mapping):
        return DiagnosticResult.failure(
            reason_code="NO_TURNOVER_EXPORT",
            message="Turnover diagnostics absent or non-numeric; skipping CSV export.",
            context={"has_risk_diag": False},
        )
    turnover_obj = diag.get("turnover")
    if isinstance(turnover_obj, pd.Series):
        series = turnover_obj.copy()
    elif isinstance(turnover_obj, Mapping):
        series = pd.Series(turnover_obj)
    elif isinstance(turnover_obj, (list, tuple)):
        series = pd.Series(turnover_obj)
    else:
        return DiagnosticResult.failure(
            reason_code="NO_TURNOVER_EXPORT",
            message="Turnover diagnostics absent or non-numeric; skipping CSV export.",
            context={"turnover_type": type(turnover_obj).__name__},
        )
    try:
        series = series.astype(float)
    except (TypeError, ValueError):
        return DiagnosticResult.failure(
            reason_code="NO_TURNOVER_EXPORT",
            message="Turnover diagnostics absent or non-numeric; skipping CSV export.",
            context={"turnover_type": type(turnover_obj).__name__},
        )
    if series.empty:
        return DiagnosticResult.failure(
            reason_code="NO_TURNOVER_EXPORT",
            message="Turnover diagnostics absent or non-numeric; skipping CSV export.",
            context={"turnover_type": type(turnover_obj).__name__},
        )
    series = series.sort_index()
    frame = series.rename("turnover").to_frame()
    frame.index.name = "Date"
    path = directory / "turnover.csv"
    frame.to_csv(path)
    return DiagnosticResult.success(path)


def _portfolio_settings(cfg: Any) -> Mapping[str, Any]:
    portfolio = getattr(cfg, "portfolio", None)
    if isinstance(portfolio, Mapping):
        return portfolio
    attrs = getattr(portfolio, "__dict__", None)
    if isinstance(attrs, Mapping):
        return cast(Mapping[str, Any], attrs)
    return {}


def _require_transaction_cost_controls(cfg: Any) -> None:
    portfolio = _portfolio_settings(cfg)
    cost_model = portfolio.get("cost_model")
    if not isinstance(cost_model, Mapping):
        raise TrendCLIError("Configuration must define portfolio.cost_model for honest costs.")
    for key in ("per_trade_bps", "half_spread_bps"):
        value = cost_model.get(key)
        if value is None:
            raise TrendCLIError(f"Configuration must define portfolio.cost_model.{key}.")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise TrendCLIError(f"portfolio.cost_model.{key} must be numeric") from exc
        if not math.isfinite(parsed):
            raise TrendCLIError(f"portfolio.cost_model.{key} must be a finite number")
        if parsed < 0:
            raise TrendCLIError(f"portfolio.cost_model.{key} cannot be negative")


def _persist_turnover_ledger(run_id: str, details: Any) -> DiagnosticResult[Path]:
    if not isinstance(details, Mapping):
        return DiagnosticResult.failure(
            reason_code="NO_TURNOVER_LEDGER",
            message="No turnover diagnostics captured for ledger persistence.",
            context={"details_type": type(details).__name__},
        )
    diag = details.get("risk_diagnostics")
    if not isinstance(diag, Mapping):
        return DiagnosticResult.failure(
            reason_code="NO_TURNOVER_LEDGER",
            message="No turnover diagnostics captured for ledger persistence.",
            context={"has_risk_diag": False},
        )
    turnover_obj = diag.get("turnover")
    if turnover_obj is None:
        return DiagnosticResult.failure(
            reason_code="NO_TURNOVER_LEDGER",
            message="No turnover diagnostics captured for ledger persistence.",
            context={"turnover_type": None},
        )
    if isinstance(turnover_obj, pd.Series):
        if turnover_obj.empty:
            return DiagnosticResult.failure(
                reason_code="NO_TURNOVER_LEDGER",
                message="No turnover diagnostics captured for ledger persistence.",
                context={"turnover_type": "Series"},
            )
    elif isinstance(turnover_obj, Mapping):
        if not turnover_obj:
            return DiagnosticResult.failure(
                reason_code="NO_TURNOVER_LEDGER",
                message="No turnover diagnostics captured for ledger persistence.",
                context={"turnover_type": "Mapping"},
            )
    elif isinstance(turnover_obj, (list, tuple)):
        if not turnover_obj:
            return DiagnosticResult.failure(
                reason_code="NO_TURNOVER_LEDGER",
                message="No turnover diagnostics captured for ledger persistence.",
                context={"turnover_type": "Sequence"},
            )
    else:
        return DiagnosticResult.failure(
            reason_code="NO_TURNOVER_LEDGER",
            message="No turnover diagnostics captured for ledger persistence.",
            context={"turnover_type": type(turnover_obj).__name__},
        )
    target_dir = Path("perf") / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    path_result = _maybe_write_turnover_csv(target_dir, details)
    if path_result.value is not None:
        print(f"Turnover ledger written to {path_result.value}")
    if path_result.diagnostic or path_result.value is None:
        return path_result
    return DiagnosticResult.success(path_result.value)
