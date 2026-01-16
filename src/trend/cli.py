from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Protocol, cast

import numpy as np
import pandas as pd
import yaml

from trend.config_schema import CoreConfigError, load_core_config
from trend.diagnostics import DiagnosticPayload, DiagnosticResult
from trend.reporting import generate_unified_report
from trend.reporting.quick_summary import main as quick_summary_main
from trend_analysis import export
from trend_analysis import logging as run_logging
from trend_analysis.api import RunResult, run_simulation
from trend_analysis.config import (
    DEFAULTS,
    ConfigPatch,
    diff_configs,
    format_validation_messages,
    validate_config,
)
from trend_analysis.config import (
    apply_patch as apply_config_patch,
)
from trend_analysis.config import load as load_config
from trend_analysis.config.coverage import (
    ConfigCoverageTracker,
    activate_config_coverage,
    deactivate_config_coverage,
    wrap_config_for_coverage,
)
from trend_analysis.config.schema_validation import load_config as load_schema_config
from trend_analysis.config.validation import ValidationResult
from trend_analysis.constants import DEFAULT_OUTPUT_DIRECTORY, DEFAULT_OUTPUT_FORMATS
from trend_analysis.data import load_csv
from trend_analysis.llm import (
    ConfigPatchChain,
    LLMProviderConfig,
    ResultSummaryChain,
    build_config_patch_prompt,
    build_result_summary_prompt,
    create_llm,
    detect_result_hallucinations,
    ensure_result_disclaimer,
    extract_metric_catalog,
    format_metric_catalog,
)
from trend_analysis.llm.nl_logging import NLOperationLog, write_nl_log
from trend_analysis.llm.replay import ReplayResult
from trend_analysis.llm.schema import load_compact_schema
from trend_analysis.logging_setup import setup_logging
from trend_model.spec import ensure_run_spec
from utils.paths import proj_path

LegacyExtractCacheStats = Callable[[object], dict[str, int] | None]

if TYPE_CHECKING:
    from trend_analysis.llm.nl_logging import NLOperationLog
    from trend_analysis.llm.replay import ReplayResult


class LegacyMaybeLogStep(Protocol):
    def __call__(self, enabled: bool, run_id: str, event: str, message: str, **fields: Any) -> None:
        # Protocol method intentionally empty; implementors provide behaviour.
        ...


def _noop_maybe_log_step(
    enabled: bool, run_id: str, event: str, message: str, **fields: Any
) -> None:
    return None


_legacy_cli_module: ModuleType | None = None
_legacy_extract_cache_stats: LegacyExtractCacheStats | None = None
_legacy_maybe_log_step: LegacyMaybeLogStep = _noop_maybe_log_step
_ORIGINAL_FALLBACKS: dict[str, Callable[..., Any]] = {}
_LEGACY_BASELINES: dict[str, Callable[..., Any]] = {}


def _report_legacy_pipeline_diagnostic(
    diagnostic: DiagnosticPayload,
    *,
    structured_log: bool,
    run_id: str,
) -> None:
    """Surface pipeline diagnostics within the legacy CLI."""

    context = diagnostic.context or {}
    text = f"Pipeline skipped ({diagnostic.reason_code}): {diagnostic.message}"
    print(text)
    safe_fields = {k: v for k, v in context.items() if isinstance(k, str)}
    _legacy_maybe_log_step(
        structured_log,
        run_id,
        "pipeline_diagnostic",
        diagnostic.message,
        reason_code=diagnostic.reason_code,
        **safe_fields,
    )


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


_USE_LEGACY_CLI = _env_flag("TREND_FORCE_LEGACY_CLI")


def _capture_legacy_baseline(name: str) -> None:
    module = _refresh_legacy_cli_module()
    if module is None or name in _LEGACY_BASELINES:
        return
    attr = getattr(module, name, None)
    if callable(attr):
        _LEGACY_BASELINES[name] = attr


def _register_fallback(name: str, fn: Callable[..., Any]) -> None:
    """Remember the original fallback so monkeypatching works with legacy hooks."""

    _ORIGINAL_FALLBACKS.setdefault(name, fn)
    _capture_legacy_baseline(name)


logger = logging.getLogger(__name__)


@dataclass
class _PerfLoggerState:
    last_path: Path | None = None
    diagnostic: DiagnosticPayload | None = None


_PERF_LOG_STATE = _PerfLoggerState()


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


def _refresh_legacy_cli_module() -> ModuleType | None:
    """Return the legacy CLI module, refreshing cached helpers when reloaded."""

    global _legacy_cli_module, _legacy_extract_cache_stats, _legacy_maybe_log_step

    module = sys.modules.get("trend_analysis.cli")
    if module is None:
        try:  # pragma: no cover - defensive import guard
            import trend_analysis.cli as module
        except Exception:  # pragma: no cover - defensive fallback
            module = None

    if module is not None and module is not _legacy_cli_module:
        _legacy_cli_module = module
        maybe_log_step_fn = getattr(module, "maybe_log_step", None)
        if callable(maybe_log_step_fn):
            _legacy_maybe_log_step = cast(LegacyMaybeLogStep, maybe_log_step_fn)
        _legacy_extract_cache_stats = getattr(module, "_extract_cache_stats", None)
        for name in _ORIGINAL_FALLBACKS:
            attr = getattr(module, name, None)
            if callable(attr) and name not in _LEGACY_BASELINES:
                _LEGACY_BASELINES[name] = attr

    return module or _legacy_cli_module


_refresh_legacy_cli_module()


APP_PATH = Path(__file__).resolve().parents[2] / "streamlit_app" / "app.py"

DEFAULT_REPORT_FORMATS = ("csv", "json", "xlsx", "txt")

SCENARIO_WINDOWS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "2008": (("2006-01", "2007-12"), ("2008-01", "2009-12")),
    "2020": (("2018-01", "2019-12"), ("2020-01", "2021-12")),
}


def _legacy_callable(name: str, fallback: Callable[..., Any]) -> Callable[..., Any]:
    original = _ORIGINAL_FALLBACKS.get(name)
    if original is not None and fallback is not original:
        return fallback
    module = _refresh_legacy_cli_module()
    if module is not None:
        attr = getattr(module, name, None)
        if callable(attr):
            baseline = _LEGACY_BASELINES.get(name)
            if baseline is None:
                _LEGACY_BASELINES[name] = attr
                baseline = attr
            if _USE_LEGACY_CLI or attr is not baseline:
                return cast(Callable[..., Any], attr)
    return fallback


class TrendCLIError(RuntimeError):
    """Raised when CLI validation fails before dispatching work."""


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the unified ``trend`` command."""

    parser = argparse.ArgumentParser(prog="trend")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    run_p = sub.add_parser("run", help="Execute the analysis pipeline")
    run_p.add_argument("-c", "--config", help="Path to YAML config")
    run_p.add_argument("--returns", help="Override returns CSV path")
    run_p.add_argument("--seed", type=int, help="Force random seed for the run")
    run_p.add_argument(
        "--bundle",
        nargs="?",
        const="analysis_bundle.zip",
        help="Write reproducibility bundle (optional path)",
    )
    run_p.add_argument("--log-file", help="Explicit JSONL log file path")
    run_p.add_argument(
        "--no-structured-log",
        action="store_true",
        help="Disable JSONL structured logging",
    )
    run_p.add_argument(
        "--config-coverage",
        action="store_true",
        help="Report which config keys were validated vs read",
    )

    report_p = sub.add_parser("report", help="Generate summary artefacts for a configuration")
    report_p.add_argument("-c", "--config", help="Path to YAML config")
    report_p.add_argument("--returns", help="Override returns CSV path")
    report_p.add_argument(
        "--out",
        help="Directory where summary outputs will be written",
    )
    report_p.add_argument(
        "--output",
        help="Path to the unified HTML report (file or directory)",
    )
    report_p.add_argument(
        "--formats",
        nargs="+",
        choices=DEFAULT_REPORT_FORMATS,
        help="Subset of export formats (default: csv json xlsx txt)",
    )
    report_p.add_argument(
        "--pdf",
        action="store_true",
        help="Also generate a PDF report alongside the HTML output",
    )
    report_p.add_argument(
        "--config-coverage",
        action="store_true",
        help="Report which config keys were validated vs read",
    )

    stress_p = sub.add_parser("stress", help="Run the pipeline against a canned stress scenario")
    stress_p.add_argument("-c", "--config", help="Path to YAML config")
    stress_p.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_WINDOWS),
        help="Stress scenario identifier",
    )
    stress_p.add_argument("--returns", help="Override returns CSV path")
    stress_p.add_argument(
        "--out",
        help="Optional export directory for stress results",
    )
    stress_p.add_argument(
        "--config-coverage",
        action="store_true",
        help="Report which config keys were validated vs read",
    )

    sub.add_parser("app", help="Launch the Streamlit application")

    quick_p = sub.add_parser("quick-report", help="Build a compact HTML report from run artefacts")
    quick_p.add_argument("--run-id", help="Run identifier (defaults to artefact inference)")
    quick_p.add_argument(
        "--artifacts",
        type=Path,
        help="Directory containing metrics_<run-id>.csv and details_<run-id>.json",
    )
    quick_p.add_argument(
        "--base-dir",
        type=Path,
        help="Base directory for derived artefacts (default: ./perf)",
    )
    quick_p.add_argument(
        "--config",
        type=Path,
        help="Configuration file to embed in the report",
    )
    quick_p.add_argument(
        "--output",
        type=Path,
        help="Explicit HTML output path (default: <base-dir>/reports/<run-id>.html)",
    )

    explain_p = sub.add_parser(
        "explain",
        help="Explain analysis results using natural language with citations",
    )
    explain_p.add_argument(
        "--details",
        type=Path,
        help="Path to details_<run-id>.json produced by the report command",
    )
    explain_p.add_argument(
        "--run-id",
        help="Run identifier used to locate details_<run-id>.json",
    )
    explain_p.add_argument(
        "--artifacts",
        type=Path,
        help="Directory containing details_<run-id>.json (default: perf)",
    )
    explain_p.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="Question to answer (repeatable; defaults to a summary prompt)",
    )
    explain_p.add_argument(
        "--questions-file",
        type=Path,
        help="Optional file containing questions (one per line)",
    )
    explain_p.add_argument(
        "--provider",
        help=(
            "LLM provider for result explanations (defaults to TREND_LLM_PROVIDER). "
            "Example: --provider openai"
        ),
    )

    nl_p = sub.add_parser("nl", help="Edit config using natural language")
    nl_p.add_argument("instruction", help="Natural language instruction to apply")
    nl_p.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        help=(
            "Input configuration file (default: config/defaults.yml) used as the base for edits. "
            "Example: --in config/base.yml"
        ),
    )
    nl_p.add_argument(
        "--out",
        dest="output_path",
        type=Path,
        help=(
            "Output configuration file (default: same as --in) for writing the updated config. "
            "Example: --out config/updated.yml"
        ),
    )
    nl_p.add_argument(
        "--diff",
        action="store_true",
        help=(
            "Print the unified diff between input and updated config without writing the file. "
            'Example: trend nl "Lower max weight" --diff'
        ),
    )
    nl_p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the updated config to stdout without writing the file. "
            'Example: trend nl "Lower max weight" --dry-run'
        ),
    )
    nl_p.add_argument(
        "--run",
        action="store_true",
        help=(
            "Validate the updated config against the schema and run the pipeline if valid. "
            'Example: trend nl "Add CSV path" --run'
        ),
    )
    nl_p.add_argument(
        "--no-confirm",
        action="store_true",
        help=(
            "Apply risky changes without an interactive confirmation prompt. "
            'Example: trend nl "Remove constraints" --no-confirm'
        ),
    )
    nl_p.add_argument(
        "--provider",
        help=(
            "LLM provider for natural language edits (defaults to TREND_LLM_PROVIDER). "
            "Example: --provider openai"
        ),
    )
    nl_p.add_argument(
        "--explain",
        action="store_true",
        help=(
            "Print an explanation of the generated changes alongside optional diff output. "
            'Example: trend nl "Lower max weight" --explain --diff'
        ),
    )

    return parser


def _resolve_returns_path(config_path: Path, cfg: Any, override: str | None) -> Path:
    """Resolve the returns CSV path relative to sensible anchors.

    Relative paths from the configuration are first checked relative to the
    configuration file itself, then the directory *above* it (repo root), and
    finally against the repository root.  This mirrors the
    ``DataSettings`` resolver so configs can reference ``demo/demo_returns.csv``
    even though the YAML file lives under ``config/``.
    """

    def _resolve_relative(raw: Path, *, include_config_roots: bool) -> Path:
        if raw.is_absolute():
            return raw.resolve()
        roots: list[Path] = []
        if include_config_roots:
            cfg_dir = config_path.parent
            roots.append(cfg_dir)
            parent = cfg_dir.parent
            if parent != cfg_dir:
                roots.append(parent)
        roots.append(proj_path())
        seen: set[Path] = set()
        for root in roots:
            if root in seen:
                continue
            seen.add(root)
            candidate = (root / raw).resolve()
            if candidate.exists():
                return candidate
        anchor = roots[0]
        return (anchor / raw).resolve()

    if override:
        return _resolve_relative(Path(override), include_config_roots=False)

    csv_path = cfg.data.get("csv_path") if hasattr(cfg, "data") else None
    if not csv_path:
        msg = "Configuration must define data.csv_path or use --returns"
        raise TrendCLIError(msg)
    return _resolve_relative(Path(csv_path), include_config_roots=True)


_register_fallback("_resolve_returns_path", _resolve_returns_path)


def _ensure_dataframe(path: Path) -> pd.DataFrame:
    try:
        df = load_csv(str(path), errors="raise")
    except TypeError:
        df = load_csv(str(path))
    if df is None:
        raise FileNotFoundError(str(path))
    return df


_register_fallback("_ensure_dataframe", _ensure_dataframe)


def _determine_seed(cfg: Any, override: int | None) -> int:
    if override is not None:
        seed = int(override)
    else:
        env_seed = os.getenv("TREND_SEED")
        if env_seed is not None:
            try:
                seed = int(env_seed)
            except (ValueError, TypeError):
                seed = getattr(cfg, "seed", 42)
        else:
            seed = getattr(cfg, "seed", 42)
    try:
        setattr(cfg, "seed", seed)
    except Exception:
        pass
    return seed


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
    except Exception:
        pass


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
    run_id = getattr(cfg, "run_id", None) or uuid.uuid4().hex[:12]
    try:
        setattr(cfg, "run_id", run_id)
    except Exception:
        pass

    log_path = None
    if structured_log:
        log_path = log_file or run_logging.get_default_log_path(run_id)
        run_logging.init_run_logger(run_id, log_path)
    _legacy_maybe_log_step(structured_log, run_id, "start", "trend CLI execution started")

    result = run_simulation(cfg, returns_df)
    diagnostic = getattr(result, "diagnostic", None)
    if diagnostic and not result.details:
        _report_legacy_pipeline_diagnostic(
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
            portfolio_series = (
                details.get("portfolio_user_weight")
                or details.get("portfolio_equal_weight")
                or details.get("portfolio_equal_weight_combined")
            )
            if portfolio_series is not None:
                setattr(result, "portfolio", portfolio_series)
        benchmarks = details.get("benchmarks")
        if isinstance(benchmarks, dict) and benchmarks:
            first = next(iter(benchmarks.values()))
            setattr(result, "benchmark", first)
        weights_user = details.get("weights_user_weight")
        if weights_user is not None:
            setattr(result, "weights", weights_user)

    _legacy_maybe_log_step(
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


_register_fallback("_run_pipeline", _run_pipeline)


def _handle_exports(cfg: Any, result: RunResult, structured_log: bool, run_id: str) -> None:
    export_cfg = getattr(cfg, "export", {}) or {}
    out_dir = export_cfg.get("directory")
    out_formats = export_cfg.get("formats")
    filename = export_cfg.get("filename", "analysis")
    if not out_dir and not out_formats:
        out_dir = DEFAULT_OUTPUT_DIRECTORY
        out_formats = DEFAULT_OUTPUT_FORMATS
    if not out_dir or not out_formats:
        return
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    data = {"metrics": result.metrics}
    split = getattr(cfg, "sample_split", {})
    in_start = str(split.get("in_start")) if split else ""
    in_end = str(split.get("in_end")) if split else ""
    out_start = str(split.get("out_start")) if split else ""
    out_end = str(split.get("out_end")) if split else ""
    if any(fmt.lower() in {"excel", "xlsx"} for fmt in out_formats):
        formatter = export.make_summary_formatter(
            result.details, in_start, in_end, out_start, out_end
        )
        data["summary"] = export.summary_frame_from_result(result.details)
        export.export_to_excel(
            data,
            str(out_dir_path / f"{filename}.xlsx"),
            default_sheet_formatter=formatter,
        )
        remaining = [fmt for fmt in out_formats if fmt.lower() not in {"excel", "xlsx"}]
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
            formats=out_formats,
        )
    _legacy_maybe_log_step(structured_log, run_id, "export_complete", "Export done")


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
    # Attach metadata expected by export_bundle
    setattr(result, "config", getattr(cfg, "__dict__", {}))
    if source_path is not None:
        setattr(result, "input_path", source_path)
    export_bundle(result, bundle_path)
    print(f"Bundle written: {bundle_path}")
    _legacy_maybe_log_step(
        structured_log,
        run_id,
        "bundle_complete",
        "Reproducibility bundle created",
        bundle=str(bundle_path),
    )


def _print_summary(cfg: Any, result: RunResult) -> None:
    split = getattr(cfg, "sample_split", {})
    text = export.format_summary_text(
        result.details,
        str(split.get("in_start", "")),
        str(split.get("in_end", "")),
        str(split.get("out_start", "")),
        str(split.get("out_end", "")),
    )
    print(text)
    if _legacy_extract_cache_stats is not None:
        cache_stats = _legacy_extract_cache_stats(result.details)
        if cache_stats:
            print("\nCache statistics:")
            for key, value in cache_stats.items():
                print(f"  {key.capitalize()}: {value}")


_register_fallback("_print_summary", _print_summary)


def _write_report_files(out_dir: Path, cfg: Any, result: RunResult, *, run_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"metrics_{run_id}.csv"
    result.metrics.to_csv(metrics_path)
    summary_path = out_dir / f"summary_{run_id}.txt"
    split = getattr(cfg, "sample_split", {})
    summary_text = export.format_summary_text(
        result.details,
        str(split.get("in_start", "")),
        str(split.get("in_end", "")),
        str(split.get("out_start", "")),
        str(split.get("out_end", "")),
    )
    summary_path.write_text(summary_text, encoding="utf-8")
    details_path = out_dir / f"details_{run_id}.json"
    with details_path.open("w", encoding="utf-8") as fh:
        json.dump(result.details, fh, default=_json_default, indent=2)
    turnover_csv_result = _maybe_write_turnover_csv(out_dir, getattr(result, "details", {}))
    if turnover_csv_result.diagnostic:
        logger.info(turnover_csv_result.diagnostic.message)
    print(f"Report artefacts written to {out_dir}")


_register_fallback("_write_report_files", _write_report_files)


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
    if isinstance(obj, pd.Series):
        data: dict[str | int | float, Any] = {}
        for key, value in obj.items():
            coerced_key: str | int | float
            if isinstance(key, (str, int, float)):
                coerced_key = key
            else:
                coerced_key = str(key)
            if isinstance(value, (np.floating, np.integer)):
                data[coerced_key] = float(value)
            else:
                data[coerced_key] = value
        return data
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    if isinstance(obj, Path):
        return str(obj)
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
    cost_value = portfolio.get("transaction_cost_bps")
    cost_model = portfolio.get("cost_model")
    if isinstance(cost_model, Mapping):
        override = cost_model.get("bps_per_trade")
        if override is not None:
            cost_value = override
        slippage = cost_model.get("slippage_bps")
        if slippage is not None:
            try:
                slip_value = float(slippage)
            except (TypeError, ValueError) as exc:
                raise TrendCLIError("portfolio.cost_model.slippage_bps must be numeric") from exc
            if slip_value < 0:
                raise TrendCLIError("portfolio.cost_model.slippage_bps cannot be negative")
    if cost_value is None:
        raise TrendCLIError(
            "Configuration must define portfolio.transaction_cost_bps for honest costs."
        )
    try:
        cost = float(cost_value)
    except (TypeError, ValueError) as exc:
        raise TrendCLIError("portfolio.transaction_cost_bps must be numeric") from exc
    if cost < 0:
        raise TrendCLIError("portfolio.transaction_cost_bps cannot be negative")


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


def _adjust_for_scenario(cfg: Any, scenario: str) -> None:
    window = SCENARIO_WINDOWS.get(scenario)
    if not window:
        raise TrendCLIError(f"Unsupported stress scenario: {scenario}")
    in_window, out_window = window
    split = dict(getattr(cfg, "sample_split", {}) or {})
    split.update(
        {
            "in_start": in_window[0],
            "in_end": in_window[1],
            "out_start": out_window[0],
            "out_end": out_window[1],
        }
    )
    try:
        setattr(cfg, "sample_split", split)
    except Exception:
        pass


def _load_configuration(path: str) -> Any:
    cfg_path = Path(path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    try:
        payload = load_schema_config(cfg_path)
    except Exception as exc:
        raise TrendCLIError(str(exc)) from exc
    try:
        load_core_config(cfg_path)
    except CoreConfigError as exc:
        raise TrendCLIError(str(exc)) from exc
    validation = validate_config(payload, base_path=cfg_path.parent, skip_required_fields=True)
    if not validation.valid:
        details = "\n".join(format_validation_messages(validation))
        raise TrendCLIError(f"Config validation failed:\n{details}")
    cfg = load_config(cfg_path)
    ensure_run_spec(cfg, base_path=cfg_path.parent)
    return cfg_path, cfg


_register_fallback("_load_configuration", _load_configuration)


def _resolve_explain_details_path(args: argparse.Namespace) -> Path:
    if args.details:
        return Path(args.details)
    if not args.run_id:
        raise TrendCLIError("The explain command requires --details or --run-id.")
    artifacts_dir = Path(args.artifacts) if args.artifacts else Path("perf")
    return artifacts_dir / f"details_{args.run_id}.json"


def _load_explain_details(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise TrendCLIError(f"Details file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrendCLIError(f"Details file is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise TrendCLIError("Details file must contain a JSON object at the root.")
    return payload


def _render_analysis_output(details: Mapping[str, Any]) -> str:
    parts: list[str] = []
    summary = pd.DataFrame()
    try:
        summary = export.summary_frame_from_result(details)
    except Exception:
        summary = pd.DataFrame()
    if not summary.empty:
        parts.append("Summary table:\n" + summary.to_string(index=False))
    else:
        parts.append("Summary table unavailable.")
    sections = ", ".join(sorted(str(k) for k in details.keys()))
    if sections:
        parts.append(f"Available sections: {sections}")
    return "\n\n".join(parts)


def _resolve_explain_questions(args: argparse.Namespace) -> str:
    questions: list[str] = []
    if args.questions:
        questions.extend([q.strip() for q in args.questions if q and q.strip()])
    if args.questions_file:
        if not args.questions_file.exists():
            raise TrendCLIError(f"Questions file not found: {args.questions_file}")
        raw_lines = args.questions_file.read_text(encoding="utf-8").splitlines()
        questions.extend([line.strip() for line in raw_lines if line.strip()])
    if not questions:
        questions = ["Summarize key findings and notable risks in the results."]
    return "\n".join(f"- {question}" for question in questions)


def _fallback_explanation(metric_catalog: str) -> str:
    if metric_catalog:
        return (
            "Unable to verify the generated explanation against the available metrics. "
            "Here is the metric catalog:\n"
            f"{metric_catalog}"
        )
    return "No metrics were detected in the analysis output."


def _resolve_llm_provider_config(provider: str | None = None) -> LLMProviderConfig:
    provider_name = (provider or os.environ.get("TREND_LLM_PROVIDER") or "openai").lower()
    supported = {"openai", "anthropic", "ollama"}
    if provider_name not in supported:
        raise TrendCLIError(
            f"Unknown LLM provider '{provider_name}'. Expected one of: {', '.join(sorted(supported))}."
        )
    api_key = os.environ.get("TREND_LLM_API_KEY")
    if not api_key:
        if provider_name == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider_name == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("TREND_LLM_MODEL")
    base_url = os.environ.get("TREND_LLM_BASE_URL")
    organization = os.environ.get("TREND_LLM_ORG")
    max_retries = os.environ.get("TREND_LLM_MAX_RETRIES")
    timeout = os.environ.get("TREND_LLM_TIMEOUT")
    max_retries_value: int | None = None
    timeout_value: float | None = None
    if max_retries:
        try:
            max_retries_value = int(max_retries)
        except ValueError as exc:
            raise TrendCLIError("TREND_LLM_MAX_RETRIES must be an integer") from exc
    if timeout:
        try:
            timeout_value = float(timeout)
        except ValueError as exc:
            raise TrendCLIError("TREND_LLM_TIMEOUT must be a number") from exc
    config_kwargs: dict[str, Any] = {"provider": provider_name}
    if model:
        config_kwargs["model"] = model
    if api_key:
        config_kwargs["api_key"] = api_key
    if base_url:
        config_kwargs["base_url"] = base_url
    if organization:
        config_kwargs["organization"] = organization
    if max_retries_value is not None:
        config_kwargs["max_retries"] = max_retries_value
    if timeout_value is not None:
        config_kwargs["timeout"] = timeout_value
    return LLMProviderConfig(**config_kwargs)


def _build_nl_chain(provider: str | None = None) -> ConfigPatchChain:
    config = _resolve_llm_provider_config(provider)
    try:
        llm = create_llm(config)
        schema = load_compact_schema()
    except Exception as exc:
        raise TrendCLIError(str(exc)) from exc
    return ConfigPatchChain.from_env(
        llm=llm,
        schema=schema,
        prompt_builder=build_config_patch_prompt,
    )


def _build_result_chain(provider: str | None = None) -> ResultSummaryChain:
    config = _resolve_llm_provider_config(provider)
    try:
        llm = create_llm(config)
    except Exception as exc:
        raise TrendCLIError(str(exc)) from exc
    return ResultSummaryChain.from_env(
        llm=llm,
        prompt_builder=build_result_summary_prompt,
    )


def _load_nl_log_entry(path: Path, entry: int) -> NLOperationLog:
    from trend_analysis.llm.replay import load_nl_log_entry

    return load_nl_log_entry(path, entry)


def _replay_nl_entry(
    entry: NLOperationLog,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> ReplayResult:
    from trend_analysis.llm.replay import replay_nl_entry

    return replay_nl_entry(entry, provider=provider, model=model, temperature=temperature)


def _build_nl_replay_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trend nl replay",
        description="Replay a logged NL operation entry.",
    )
    parser.add_argument("log_file", type=Path, help="Path to nl_ops_<date>.jsonl log file")
    parser.add_argument("--entry", type=int, required=True, help="1-based entry index")
    parser.add_argument("--provider", help="Override the logged LLM provider")
    parser.add_argument("--model", help="Override the logged LLM model")
    parser.add_argument("--temperature", type=float, help="Override the logged temperature")
    parser.add_argument("--show-prompt", action="store_true", help="Print the prompt text")
    return parser


def _run_nl_replay(argv: list[str]) -> int:
    parser = _build_nl_replay_parser()
    args = parser.parse_args(argv)
    log_path = Path(args.log_file)
    if not log_path.exists():
        raise TrendCLIError(f"Log file not found: {log_path}")
    try:
        entry = _load_nl_log_entry(log_path, args.entry)
    except (ValueError, IndexError) as exc:
        raise TrendCLIError(str(exc)) from exc
    result = _replay_nl_entry(
        entry,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )
    if args.show_prompt:
        print("Prompt:")
        print(result.prompt)
    print(f"Prompt hash: {result.prompt_hash}")
    print(f"Output hash: {result.output_hash}")
    if result.trace_url:
        print(f"Trace URL: {result.trace_url}")
    if result.recorded_hash is None:
        print("Recorded hash: <none>")
    else:
        print(f"Recorded hash: {result.recorded_hash}")
    print(f"Matches: {result.matches}")
    if result.recorded_output is None:
        print("Recorded output: <none>")
    else:
        print("Recorded output:")
        print(result.recorded_output)
    if result.recorded_output is None:
        print("Comparison: skipped (no recorded output)")
        exit_code = 0
    elif result.matches:
        print("Comparison: match")
        exit_code = 0
    else:
        print("Comparison: mismatch")
        exit_code = 1
    if result.diff:
        print("Diff:")
        print(result.diff)
    print("Replay output:")
    print(result.output)
    return exit_code


def _maybe_handle_nl_replay(argv: list[str]) -> int | None:
    if len(argv) >= 2 and argv[0] == "nl" and argv[1] == "replay":
        return _run_nl_replay(argv[2:])
    return None


def _load_nl_config(path: Path) -> dict[str, Any]:
    try:
        payload = load_schema_config(path)
    except Exception as exc:
        raise TrendCLIError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise TrendCLIError("Config file must contain a mapping at the root.")
    return payload


def _hash_nl_payload(payload: dict[str, Any]) -> str:
    text = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _log_nl_operation(
    *,
    request_id: str,
    operation: str,
    input_payload: dict[str, Any],
    model_name: str,
    temperature: float,
    parsed_patch: ConfigPatch | None = None,
    validation_result: ValidationResult | None = None,
    error: str | None = None,
    started_at: float,
    timestamp: datetime,
) -> None:
    entry = NLOperationLog(
        request_id=request_id,
        timestamp=timestamp,
        operation=cast(Any, operation),
        input_hash=_hash_nl_payload(input_payload),
        prompt_template="",
        prompt_variables={},
        model_output=None,
        parsed_patch=parsed_patch,
        validation_result=validation_result,
        error=error,
        duration_ms=(time.perf_counter() - started_at) * 1000,
        model_name=model_name,
        temperature=temperature,
        token_usage=None,
    )
    write_nl_log(entry)


def _apply_nl_instruction(
    config: dict[str, Any],
    instruction: str,
    *,
    provider: str | None = None,
    request_id: str,
) -> tuple[ConfigPatch, dict[str, Any], str, str, float]:
    chain = _build_nl_chain(provider)
    try:
        patch = chain.run(current_config=config, instruction=instruction, request_id=request_id)
    except Exception as exc:
        raise TrendCLIError(str(exc)) from exc
    apply_started = time.perf_counter()
    apply_timestamp = datetime.now(timezone.utc)
    apply_error: str | None = None
    try:
        updated = apply_config_patch(config, patch)
    except Exception as exc:
        apply_error = str(exc) or type(exc).__name__
        raise TrendCLIError(str(exc)) from exc
    finally:
        _log_nl_operation(
            request_id=request_id,
            operation="apply_patch",
            input_payload={
                "config": config,
                "patch": patch.model_dump(mode="json"),
            },
            model_name=chain.model or "unknown",
            temperature=chain.temperature,
            parsed_patch=patch,
            error=apply_error,
            started_at=apply_started,
            timestamp=apply_timestamp,
        )
    diff = diff_configs(config, updated)
    return patch, updated, diff, chain.model or "unknown", chain.temperature


def _format_nl_explanation(patch: ConfigPatch) -> str:
    lines = [f"Summary: {patch.summary}"]
    if patch.risk_flags:
        flags = ", ".join(flag.value for flag in patch.risk_flags)
        lines.append(f"Risk flags: {flags}")
    if patch.needs_review:
        lines.append("Needs review: unknown config keys detected.")
    rationales = [
        (operation.path, operation.rationale)
        for operation in patch.operations
        if operation.rationale
    ]
    if rationales:
        lines.append("Rationales:")
        lines.extend(f"- {path}: {rationale}" for path, rationale in rationales)
    return "\n".join(lines).strip() + "\n"


def _validate_nl_run_config(updated: dict[str, Any], *, base_path: Path) -> None:
    validation = validate_config(
        updated,
        base_path=base_path,
        include_model_validation=True,
    )
    if not validation.valid:
        details = "\n".join(format_validation_messages(validation))
        raise TrendCLIError(f"Config validation failed:\n{details}")


def _confirm_risky_patch(patch: ConfigPatch, *, no_confirm: bool) -> None:
    flags = [flag.value for flag in patch.risk_flags]
    if patch.needs_review:
        flags.append("UNKNOWN_KEYS")
    if not flags or no_confirm:
        return
    flags_text = ", ".join(flags)
    if not sys.stdin.isatty():
        raise TrendCLIError(
            f"Risky changes detected ({flags_text}). Re-run with --no-confirm to apply without prompting."
        )
    response = input(f"Risky changes detected ({flags_text}). Continue? [y/N]: ")
    if response.strip().lower() not in {"y", "yes"}:
        raise TrendCLIError("Update cancelled by user.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        argv_list = argv if argv is not None else sys.argv[1:]
        maybe_replay_exit = _maybe_handle_nl_replay(argv_list)
        if maybe_replay_exit is not None:
            return maybe_replay_exit
        args = parser.parse_args(argv)

        command = args.subcommand
        coverage_tracker: ConfigCoverageTracker | None = None
        if getattr(args, "config_coverage", False):
            coverage_tracker = ConfigCoverageTracker()
            activate_config_coverage(coverage_tracker)

        def _finalize_config_coverage() -> None:
            if coverage_tracker is None:
                return
            print(coverage_tracker.format_report())
            deactivate_config_coverage()

        if command == "app":
            if coverage_tracker is not None:
                deactivate_config_coverage()
            proc = subprocess.run(["streamlit", "run", str(APP_PATH)])
            return proc.returncode

        if command == "quick-report":
            if coverage_tracker is not None:
                deactivate_config_coverage()
            quick_args: list[str] = []
            if args.run_id:
                quick_args.extend(["--run-id", args.run_id])
            if args.artifacts:
                quick_args.extend(["--artifacts", os.fspath(args.artifacts)])
            if args.base_dir:
                quick_args.extend(["--base-dir", os.fspath(args.base_dir)])
            if args.config:
                quick_args.extend(["--config", os.fspath(args.config)])
            if args.output:
                quick_args.extend(["--output", os.fspath(args.output)])
            return quick_summary_main(quick_args)

        if command == "explain":
            request_id = uuid.uuid4().hex
            details_path = _resolve_explain_details_path(args)
            details = _load_explain_details(details_path)
            entries = extract_metric_catalog(details)
            metric_catalog = format_metric_catalog(entries)
            if not entries:
                explanation = ensure_result_disclaimer(
                    "No metrics were detected in the analysis output."
                )
                print(explanation)
                return 0
            analysis_output = _render_analysis_output(details)
            questions = _resolve_explain_questions(args)
            chain = _build_result_chain(args.provider)
            response = chain.run(
                analysis_output=analysis_output,
                metric_catalog=metric_catalog,
                questions=questions,
                request_id=request_id,
            )
            raw_explanation = response.text
            hallucinations = detect_result_hallucinations(
                raw_explanation,
                entries,
                logger=logger,
            )
            if hallucinations:
                raw_explanation = _fallback_explanation(metric_catalog)
            explanation = ensure_result_disclaimer(raw_explanation)
            print(explanation)
            return 0

        if command == "nl":
            request_id = uuid.uuid4().hex
            input_path = Path(args.input_path) if args.input_path else DEFAULTS
            if not input_path.exists():
                raise TrendCLIError(f"Input config not found: {input_path}")
            output_path = Path(args.output_path) if args.output_path else input_path
            config = _load_nl_config(input_path)
            patch, updated, diff, model_name, temperature = _apply_nl_instruction(
                config,
                args.instruction,
                provider=args.provider,
                request_id=request_id,
            )
            if args.run and (args.diff or args.dry_run):
                raise TrendCLIError("--run cannot be combined with --diff or --dry-run")
            if args.run:
                _validate_nl_run_config(updated, base_path=output_path.parent)
            if args.explain:
                sys.stdout.write(_format_nl_explanation(patch))
            if args.diff:
                if diff:
                    sys.stdout.write(diff)
                else:
                    print("No changes.")
                return 0
            if args.dry_run:
                sys.stdout.write(yaml.safe_dump(updated, sort_keys=False, default_flow_style=False))
                return 0
            if args.run:
                validate_started = time.perf_counter()
                validate_timestamp = datetime.now(timezone.utc)
                validation_error: str | None = None
                try:
                    validation = validate_config(
                        updated,
                        base_path=output_path.parent,
                        include_model_validation=True,
                    )
                except Exception as exc:
                    validation_error = str(exc) or type(exc).__name__
                    _log_nl_operation(
                        request_id=request_id,
                        operation="validate",
                        input_payload={
                            "config": updated,
                            "base_path": output_path.parent,
                        },
                        model_name=model_name,
                        temperature=temperature,
                        parsed_patch=patch,
                        error=validation_error,
                        started_at=validate_started,
                        timestamp=validate_timestamp,
                    )
                    raise TrendCLIError(str(exc)) from exc
                if not validation.valid:
                    details = "\n".join(format_validation_messages(validation))
                    validation_error = f"validation failed: {details}"
                _log_nl_operation(
                    request_id=request_id,
                    operation="validate",
                    input_payload={"config": updated, "base_path": output_path.parent},
                    model_name=model_name,
                    temperature=temperature,
                    parsed_patch=patch,
                    validation_result=validation,
                    error=validation_error,
                    started_at=validate_started,
                    timestamp=validate_timestamp,
                )
                if validation_error is not None:
                    raise TrendCLIError(f"Config validation failed:\n{details}")
            _confirm_risky_patch(patch, no_confirm=args.no_confirm)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                yaml.safe_dump(updated, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
            print(f"Updated config written: {output_path}")
            if args.run:
                try:
                    cfg = load_config(output_path)
                except Exception as exc:
                    raise TrendCLIError(str(exc)) from exc
                ensure_run_spec(cfg, base_path=output_path.parent)
                returns_path = _resolve_returns_path(output_path, cfg, None)
                returns_df = _ensure_dataframe(returns_path)
                _determine_seed(cfg, None)
                run_pipeline = _legacy_callable("_run_pipeline", _run_pipeline)
                run_started = time.perf_counter()
                run_timestamp = datetime.now(timezone.utc)
                run_error: str | None = None
                try:
                    result, run_id, log_path = run_pipeline(
                        cfg,
                        returns_df,
                        source_path=returns_path,
                        log_file=None,
                        structured_log=True,
                        bundle=None,
                    )
                except Exception as exc:
                    run_error = str(exc) or type(exc).__name__
                    raise TrendCLIError(str(exc)) from exc
                finally:
                    _log_nl_operation(
                        request_id=request_id,
                        operation="run",
                        input_payload={
                            "config_path": str(output_path),
                            "returns_path": str(returns_path),
                        },
                        model_name=model_name,
                        temperature=temperature,
                        parsed_patch=patch,
                        error=run_error,
                        started_at=run_started,
                        timestamp=run_timestamp,
                    )
                print_summary = _legacy_callable("_print_summary", _print_summary)
                print_summary(cfg, result)
                if log_path:
                    print(f"Structured log: {log_path}")
            return 0

        if command not in {"run", "report", "stress"}:
            raise TrendCLIError(f"Unknown command: {command}")

        if not args.config:
            raise TrendCLIError(f"The --config option is required for the '{command}' command")

        load_config_fn = _legacy_callable("_load_configuration", _load_configuration)
        cfg_path, cfg = load_config_fn(args.config)
        if coverage_tracker is not None:
            wrap_config_for_coverage(cfg, coverage_tracker)
        ensure_run_spec(cfg, base_path=cfg_path.parent)
        resolve_returns = _legacy_callable("_resolve_returns_path", _resolve_returns_path)
        returns_path = resolve_returns(cfg_path, cfg, getattr(args, "returns", None))
        ensure_df = _legacy_callable("_ensure_dataframe", _ensure_dataframe)
        returns_df = ensure_df(returns_path)
        seed = _determine_seed(cfg, getattr(args, "seed", None))

        if command == "run":
            run_pipeline = _legacy_callable("_run_pipeline", _run_pipeline)
            result, run_id, log_path = run_pipeline(
                cfg,
                returns_df,
                source_path=returns_path,
                log_file=Path(args.log_file) if args.log_file else None,
                structured_log=not args.no_structured_log,
                bundle=Path(args.bundle) if args.bundle else None,
            )
            print_summary = _legacy_callable("_print_summary", _print_summary)
            print_summary(cfg, result)
            if log_path:
                print(f"Structured log: {log_path}")
            _finalize_config_coverage()
            return 0

        if command == "report":
            export_dir = Path(args.out).resolve() if args.out else None
            if export_dir is None and not args.output:
                raise TrendCLIError(
                    "The 'report' command requires --out for artefacts or --output for the HTML report"
                )
            formats = args.formats or DEFAULT_REPORT_FORMATS
            _prepare_export_config(cfg, export_dir, formats if export_dir is not None else None)
            run_pipeline = _legacy_callable("_run_pipeline", _run_pipeline)
            result, run_id, _ = run_pipeline(
                cfg,
                returns_df,
                source_path=returns_path,
                log_file=None,
                structured_log=False,
                bundle=None,
            )
            print_summary = _legacy_callable("_print_summary", _print_summary)
            print_summary(cfg, result)
            if export_dir is not None:
                write_report = _legacy_callable("_write_report_files", _write_report_files)
                write_report(export_dir, cfg, result, run_id=run_id)
            report_path = _resolve_report_output_path(args.output, export_dir, run_id)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                artifacts = generate_unified_report(
                    result,
                    cfg,
                    run_id=run_id,
                    include_pdf=args.pdf,
                    spec=getattr(cfg, "_trend_run_spec", None),
                )
            except RuntimeError as exc:
                raise TrendCLIError(str(exc)) from exc
            report_path.write_text(artifacts.html, encoding="utf-8")
            print(f"Report written: {report_path}")
            if args.pdf:
                if artifacts.pdf_bytes is None:
                    raise TrendCLIError(
                        "PDF generation failed – install the 'fpdf2' dependency to enable --pdf output"
                    )
                pdf_path = report_path.with_suffix(".pdf")
                pdf_path.write_bytes(artifacts.pdf_bytes)
                print(f"PDF report written: {pdf_path}")
            _finalize_config_coverage()
            return 0

        if command == "stress":
            if not args.scenario:
                raise TrendCLIError("The --scenario option is required for the 'stress' command")
            _adjust_for_scenario(cfg, args.scenario)
            export_dir = Path(args.out) if args.out else None
            _prepare_export_config(cfg, export_dir, None)
            run_pipeline = _legacy_callable("_run_pipeline", _run_pipeline)
            result, run_id, _ = run_pipeline(
                cfg,
                returns_df,
                source_path=returns_path,
                log_file=None,
                structured_log=False,
                bundle=None,
            )
            print(f"Stress scenario '{args.scenario}' completed (seed={seed}).")
            print_summary = _legacy_callable("_print_summary", _print_summary)
            print_summary(cfg, result)
            if export_dir:
                write_report = _legacy_callable("_write_report_files", _write_report_files)
                write_report(export_dir, cfg, result, run_id=run_id)
            _finalize_config_coverage()
            return 0

        raise TrendCLIError(f"Unknown command: {command}")
    except TrendCLIError as exc:
        if "coverage_tracker" in locals() and coverage_tracker is not None:
            deactivate_config_coverage()
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        if "coverage_tracker" in locals() and coverage_tracker is not None:
            deactivate_config_coverage()
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
