from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
import yaml

from trend.cli_commands import (
    PreparedCommandInputs,
    prepare_command_inputs,
    run_analysis_command,
    run_app_command,
    run_check_command,
    run_report_command,
)
from trend.cli_helpers import (
    _apply_trend_spec_preset,
    _apply_universe_mask,
    _attach_universe_paths,
)
from trend.commands.explain import (
    _build_explain_artifact_payload,
    _build_result_chain,
    _fallback_explanation,
    _finalize_explanation_text,
    _infer_explain_run_id,
    _load_explain_details,
    _render_analysis_output,
    _resolve_explain_details_path,
    _resolve_explain_questions,
    _write_explain_artifacts,
)
from trend.commands.nl import (
    _apply_nl_instruction,
    _confirm_risky_patch,
    _format_nl_explanation,
    _load_nl_config,
    _log_nl_operation,
    _maybe_handle_nl_replay,
    _validate_nl_run_config,
)
from trend.commands.report_export import (
    _finish_structured_log,
    _prepare_export_config,
    _print_summary,
    _resolve_report_output_path,
    _run_pipeline,
    _write_report_files,
    _write_trend_run_artifacts,
)
from trend.cli_support import (
    check_environment,
    find_prior_run,
    maybe_log_step,
)
from trend.config_schema import CoreConfigError, load_core_config
from trend.reporting import generate_unified_report
from trend.reporting.quick_summary import main as quick_summary_main
from trend.spec import ensure_run_spec
from trend_analysis import logging as run_logging
from trend_analysis.config import (
    DEFAULTS,
    format_validation_messages,
    validate_config,
)
from trend_analysis.config import load as load_config
from trend_analysis.config.coverage import (
    ConfigCoverageTracker,
    activate_config_coverage,
    deactivate_config_coverage,
    wrap_config_for_coverage,
)
from trend_analysis.config.schema_validation import load_config as load_schema_config
from trend_analysis.config.ui_mapping import build_config_from_ui_state
from trend_analysis.data import load_csv
from trend_analysis.io.market_data import MarketDataValidationError
from trend_analysis.io.ui_ingest import inspect_ui_date_issues, load_ui_dataset
from trend_analysis.llm import (
    build_deterministic_feedback,
    compact_metric_catalog,
    extract_metric_catalog,
    format_metric_catalog,
    postprocess_result_text,
)
from trend_analysis.llm.result_validation import (
    append_discrepancy_log,
)
from trend_analysis.perf.rolling_cache import set_cache_enabled
from trend_analysis.presets import (
    apply_trend_preset,
    get_trend_preset,
    list_preset_slugs,
)
from trend_analysis.signal_presets import (
    get_trend_spec_preset,
    list_trend_spec_presets,
)
from trend_analysis.universe_catalog import load_universe
from trend_analysis.util.hash import working_run_id
from utils.paths import proj_path


def _run_environment_check() -> int:
    return check_environment()


logger = logging.getLogger(__name__)


APP_PATH = Path(__file__).resolve().parents[2] / "streamlit_app" / "app.py"

DEFAULT_REPORT_FORMATS = ("csv", "json", "xlsx", "txt")

SCENARIO_WINDOWS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "2008": (("2006-01", "2007-12"), ("2008-01", "2009-12")),
    "2020": (("2018-01", "2019-12"), ("2020-01", "2021-12")),
}


# TrendCLIError is the canonical user-facing error for all CLI validation
# failures.  It is defined in the shared ``trend.mc.viz`` module so that the
# Monte Carlo visualisation pipeline can raise the same exception type
# regardless of which CLI entry-point invoked it.
from trend.mc.commands import (  # noqa: E402
    add_mc_subparsers,
    handle_mc_command,
    is_valid_tqdm_instance,
    write_mc_manifest,
)
from trend.mc.viz import TrendCLIError  # noqa: E402


def _write_mc_manifest(*args: Any, **kwargs: Any) -> Path:
    """Compatibility test hook for the shared Monte Carlo manifest writer."""

    return write_mc_manifest(*args, **kwargs)


def _is_valid_tqdm_instance(candidate: Any) -> bool:
    """Compatibility test hook for the shared Monte Carlo progress validator."""

    return is_valid_tqdm_instance(candidate)


def build_parser(*, prog: str = "trend") -> argparse.ArgumentParser:
    """Construct the argument parser for the unified ``trend`` command."""
    parser = argparse.ArgumentParser(prog=prog)
    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("check", help="Print environment info and exit")

    run_p = sub.add_parser("run", help="Execute the analysis pipeline")
    run_p.add_argument("-c", "--config", help="Path to YAML config")
    run_p.add_argument(
        "-i",
        "--input",
        "--returns",
        dest="returns",
        help="Override returns CSV path",
    )
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
        "--universe",
        help="Select a named universe defined under config/universe",
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
        "--skip-if-exists",
        action="store_true",
        help=(
            "Reuse a prior completed run for the content-addressed run_id "
            "instead of recomputing; reports the existing manifest path"
        ),
    )
    run_p.add_argument(
        "--auto-fix-dates",
        action="store_true",
        help="Apply Streamlit-style date corrections to replay input data",
    )
    run_p.add_argument(
        "--yes",
        action="store_true",
        help="Approve Streamlit-style date corrections without prompting",
    )

    report_p = sub.add_parser("report", help="Generate summary artefacts for a configuration")
    report_p.add_argument("-c", "--config", help="Path to YAML config")
    report_p.add_argument(
        "-i",
        "--input",
        "--returns",
        dest="returns",
        help="Override returns CSV path",
    )
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

    mc_p = sub.add_parser("mc", help="Monte Carlo scenario workflows")
    add_mc_subparsers(mc_p)

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
    explain_p.add_argument(
        "--output",
        type=Path,
        help=(
            "Directory (created if needed) or file prefix for explanation artifacts. "
            "Example: --output perf/explanations or --output perf/explanation_run.txt"
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
        "--model",
        help=(
            "Override the LLM model for natural language edits (defaults to TREND_LLM_MODEL). "
            "Example: --model gpt-4o-mini"
        ),
    )
    nl_p.add_argument(
        "--temperature",
        type=float,
        help=(
            "Override the LLM temperature for natural language edits (defaults to TREND_LLM_TEMPERATURE). "
            "Example: --temperature 0.2"
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
    ``ResolvedDataSettings`` resolver so configs can reference ``demo/demo_returns.csv``
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


def _ensure_dataframe(
    path: Path,
    *,
    config: Any | None = None,
    auto_fix_dates: bool = False,
    yes: bool = False,
) -> pd.DataFrame:
    data_settings = getattr(config, "data", {}) if config is not None else {}
    if isinstance(data_settings, Mapping):
        missing_policy = data_settings.get("missing_policy")
        missing_limit = data_settings.get("missing_limit")
        date_column = data_settings.get("date_column", "Date") or "Date"
    else:
        missing_policy = getattr(data_settings, "missing_policy", None)
        missing_limit = getattr(data_settings, "missing_limit", None)
        date_column = getattr(data_settings, "date_column", "Date") or "Date"
    if auto_fix_dates:
        _confirm_ui_date_fixes(path, yes=yes, date_column=str(date_column))
        try:
            frame, _, summary = load_ui_dataset(
                path,
                auto_fix_dates=True,
                date_column=str(date_column),
                missing_policy=missing_policy or "drop",
                missing_limit=missing_limit,
            )
        except MarketDataValidationError as exc:
            details = "\n".join(f"- {issue}" for issue in exc.issues)
            suffix = f"\n{details}" if details else ""
            raise TrendCLIError(f"{exc.user_message}{suffix}") from exc
        changes = []
        if summary.corrected_dates:
            changes.append(f"{summary.corrected_dates} date correction(s)")
        if summary.dropped_rows:
            changes.append(f"{summary.dropped_rows} row(s) dropped")
        if summary.dropped_columns:
            changes.append("dropped date-named column(s): " + ", ".join(summary.dropped_columns))
        if changes:
            print(f"Applied UI-style date fixes: {', '.join(changes)}")
        return frame.rename_axis("Date").reset_index()

    try:
        df = load_csv(
            str(path),
            errors="raise",
            date_column=str(date_column),
            missing_policy=missing_policy,
            missing_limit=missing_limit,
        )
    except MarketDataValidationError as exc:
        details = "\n".join(f"- {issue}" for issue in exc.issues)
        suffix = f"\n{details}" if details else ""
        raise TrendCLIError(f"{exc.user_message}{suffix}") from exc
    if df is None:
        raise FileNotFoundError(str(path))
    return df


def _load_ui_payload(
    path: Path,
    payload_any: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Load a nested or flat Streamlit model-state export."""

    if payload_any is None:
        try:
            payload_any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrendCLIError(f"Unable to load Streamlit JSON export: {exc}") from exc
    if not isinstance(payload_any, Mapping):
        raise TrendCLIError("Streamlit JSON export must contain an object at the root")
    payload = dict(payload_any)
    model_state = payload.get("model_state")
    if isinstance(model_state, Mapping):
        return payload, model_state
    return {"model_state": payload}, payload


def _looks_like_model_state(payload: Mapping[str, Any]) -> bool:
    """Distinguish flat Streamlit exports from schema-native JSON configs."""

    ui_keys = {
        "lookback_periods",
        "evaluation_periods",
        "selection_count",
        "metric_weights",
        "signal_window",
        "signal_lag",
        "signal_min_periods",
        "signal_zscore",
        "signal_vol_adjust",
        "signal_vol_target",
        "vol_adjust_enabled",
        "risk_target",
        "multi_period_enabled",
        "multi_period_frequency",
        "start_date",
        "end_date",
        "date_mode",
    }
    schema_keys = {
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
    if any(key in payload for key in schema_keys):
        return False
    hits = sum(key in payload for key in ui_keys)
    return hits >= 3 or ("metric_weights" in payload and "selection_count" in payload)


def _read_ui_config_payload(path: Path) -> Mapping[str, Any] | None:
    if path.suffix.lower() != ".json":
        return None
    try:
        payload_any: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload_any, Mapping):
        return None
    if "model_state" in payload_any or _looks_like_model_state(payload_any):
        return payload_any
    return None


def _should_handle_as_ui_config(path: Path) -> bool:
    return _read_ui_config_payload(path) is not None


def _confirm_ui_date_fixes(
    data_path: Path,
    *,
    yes: bool,
    date_column: str = "Date",
) -> None:
    try:
        issues = inspect_ui_date_issues(data_path, date_column=date_column)
    except MarketDataValidationError as exc:
        details = "\n".join(f"- {issue}" for issue in exc.issues)
        suffix = f"\n{details}" if details else ""
        raise TrendCLIError(f"{exc.user_message}{suffix}") from exc

    has_fixable = issues.has_corrections or issues.total_droppable_rows > 0
    if not has_fixable or yes:
        return
    if not sys.stdin.isatty():
        raise TrendCLIError("Date corrections require confirmation. Re-run with --yes to approve.")
    prompt = (
        f"Apply {len(issues.corrections)} date correction(s) and "
        f"drop {issues.total_droppable_rows} row(s)? [y/N]: "
    )
    if input(prompt).strip().lower() not in {"y", "yes"}:
        raise TrendCLIError("Date corrections cancelled")


def _prepare_ui_command_inputs(
    args: argparse.Namespace,
    *,
    prepare_config: Callable[[Any], None],
    ui_payload: Mapping[str, Any] | None = None,
) -> PreparedCommandInputs:
    """Map a Streamlit export onto the canonical ``trend run`` inputs."""

    params_path = Path(args.config).resolve()
    if not args.returns:
        raise TrendCLIError("Streamlit JSON replay requires --input/--returns data")
    data_path = Path(args.returns).resolve()
    payload, model_state = _load_ui_payload(params_path, ui_payload)

    if "risk_free_column" not in model_state:
        risk_free = payload.get("selected_risk_free")
        if isinstance(risk_free, str) and risk_free.strip():
            model_state = dict(model_state)
            model_state["risk_free_column"] = risk_free.strip()

    benchmark = payload.get("selected_benchmark")
    if not benchmark:
        candidate = model_state.get("info_ratio_benchmark")
        if isinstance(candidate, str) and candidate.strip():
            benchmark = candidate.strip()

    if args.auto_fix_dates:
        _confirm_ui_date_fixes(data_path, yes=args.yes)
    try:
        returns, metadata, summary = load_ui_dataset(
            data_path,
            auto_fix_dates=args.auto_fix_dates,
            missing_policy=model_state.get("missing_policy") or "drop",
            missing_limit=model_state.get("missing_limit"),
        )
    except MarketDataValidationError as exc:
        details = "\n".join(f"- {issue}" for issue in exc.issues)
        suffix = f"\n{details}" if details else ""
        raise TrendCLIError(f"{exc.user_message}{suffix}") from exc

    if summary.corrected_dates or summary.dropped_rows or summary.dropped_columns:
        changes = []
        if summary.corrected_dates:
            changes.append(f"{summary.corrected_dates} date correction(s)")
        if summary.dropped_rows:
            changes.append(f"{summary.dropped_rows} row(s) dropped")
        if summary.dropped_columns:
            changes.append("dropped date-named column(s): " + ", ".join(summary.dropped_columns))
        print(f"Applied UI-style date fixes: {', '.join(changes)}")

    cfg = build_config_from_ui_state(
        returns=returns,
        model_state=model_state,
        benchmark=benchmark if isinstance(benchmark, str) else None,
        frequency=metadata.frequency,
        csv_path=str(data_path) if data_path.suffix.lower() == ".csv" else None,
    )
    prepare_config(cfg)
    ensure_run_spec(cfg, base_path=params_path.parent, required=True)
    seed = _determine_seed(cfg, args.seed)
    return PreparedCommandInputs(
        cfg_path=params_path,
        cfg=cfg,
        returns_path=data_path,
        returns_df=returns.reset_index(),
        seed=seed,
    )


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
    ensure_run_spec(cfg, base_path=cfg_path.parent, required=True)
    return cfg_path, cfg


def main(argv: list[str] | None = None, *, prog: str = "trend") -> int:
    try:
        parser = build_parser(prog=prog)
    except TypeError:
        parser = build_parser()
    try:
        argv_list = argv if argv is not None else sys.argv[1:]
        maybe_replay_exit = _maybe_handle_nl_replay(argv_list)
        if maybe_replay_exit is not None:
            return maybe_replay_exit
        args, extra_args = parser.parse_known_args(argv_list)

        command = args.subcommand
        if extra_args and command != "app":
            raise TrendCLIError(f"Unexpected arguments: {' '.join(extra_args)}")
        coverage_tracker: ConfigCoverageTracker | None = None
        if getattr(args, "config_coverage", False):
            coverage_tracker = ConfigCoverageTracker()
            activate_config_coverage(coverage_tracker)

        def _finalize_config_coverage() -> None:
            if coverage_tracker is None:
                return
            print(coverage_tracker.format_report())
            deactivate_config_coverage()

        if command == "check":
            if coverage_tracker is not None:
                deactivate_config_coverage()
            return run_check_command(environment_check=_run_environment_check)

        if command == "app":
            if coverage_tracker is not None:
                deactivate_config_coverage()
            return run_app_command(
                args,
                extra_args,
                app_path=APP_PATH,
                run_process=subprocess.run,
            )

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
            questions = _resolve_explain_questions(args)
            run_id = _infer_explain_run_id(details_path, args.run_id, details)
            created_at = datetime.now(timezone.utc)
            all_entries = extract_metric_catalog(details)
            compacted_entries = compact_metric_catalog(all_entries, questions=questions)
            metric_catalog = format_metric_catalog(compacted_entries)
            if not all_entries:
                explanation = _finalize_explanation_text(
                    "No metrics were detected in the analysis output.",
                    [],
                )
                if args.output:
                    payload = _build_explain_artifact_payload(
                        run_id=run_id,
                        created_at=created_at,
                        text=explanation,
                        metric_count=0,
                        trace_url=None,
                        claim_issues=[],
                        questions=questions,
                    )
                    _write_explain_artifacts(
                        output=Path(args.output),
                        run_id=run_id,
                        text=explanation,
                        payload=payload,
                    )
                else:
                    print(explanation)
                return 0
            analysis_output = _render_analysis_output(details)
            diagnostics = build_deterministic_feedback(details, all_entries)
            if diagnostics:
                analysis_output = f"{analysis_output}\n\n{diagnostics}"
            chain = _build_result_chain(args.provider)
            response = chain.run(
                analysis_output=analysis_output,
                metric_catalog=metric_catalog,
                questions=questions,
                request_id=request_id,
                metric_entries=all_entries,
            )
            explanation_text, claim_issues = postprocess_result_text(
                response.text,
                all_entries,
                logger=logger,
            )
            trace_url = getattr(response, "trace_url", None)
            if claim_issues:
                fallback = _fallback_explanation(metric_catalog)
                fallback = append_discrepancy_log(fallback, claim_issues)
                explanation_text = fallback
            explanation_text = _finalize_explanation_text(explanation_text, claim_issues)
            if args.output:
                payload = _build_explain_artifact_payload(
                    run_id=run_id,
                    created_at=created_at,
                    text=explanation_text,
                    metric_count=len(compacted_entries),
                    trace_url=trace_url,
                    claim_issues=claim_issues,
                    questions=questions,
                )
                _write_explain_artifacts(
                    output=Path(args.output),
                    run_id=run_id,
                    text=explanation_text,
                    payload=payload,
                )
            else:
                print(explanation_text)
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
                model=args.model,
                temperature=args.temperature,
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
                    validation_details = "\n".join(format_validation_messages(validation))
                    validation_error = f"validation failed: {validation_details}"
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
                    raise TrendCLIError(f"Config validation failed:\n{validation_details}")
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
                ensure_run_spec(cfg, base_path=output_path.parent, required=True)
                returns_path = _resolve_returns_path(output_path, cfg, None)
                returns_df = _ensure_dataframe(returns_path)
                _determine_seed(cfg, None)
                run_pipeline = _run_pipeline
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
                print_summary = _print_summary
                print_summary(cfg, result)
                if log_path:
                    print(f"Structured log: {log_path}")
            return 0

        if command == "mc":
            _finalize_config_coverage()
            return handle_mc_command(args)

        if command not in {"run", "report", "stress", "mc"}:
            raise TrendCLIError(f"Unknown command: {command}")

        if command != "mc" and not args.config:
            raise TrendCLIError(f"The --config option is required for the '{command}' command")

        def _prepare_config_for_command(cfg: Any) -> None:
            if coverage_tracker is not None:
                wrap_config_for_coverage(cfg, coverage_tracker)

        config_path = Path(args.config).resolve()
        ui_payload = _read_ui_config_payload(config_path) if command == "run" else None
        if command == "run" and ui_payload is not None:
            prepared = _prepare_ui_command_inputs(
                args,
                prepare_config=_prepare_config_for_command,
                ui_payload=ui_payload,
            )
        else:
            prepared = prepare_command_inputs(
                args,
                load_configuration=_load_configuration,
                prepare_config=_prepare_config_for_command,
                ensure_run_spec=ensure_run_spec,
                resolve_returns_path=_resolve_returns_path,
                ensure_dataframe=_ensure_dataframe,
                determine_seed=_determine_seed,
            )
        cfg_path = prepared.cfg_path
        cfg = prepared.cfg
        returns_path = prepared.returns_path
        returns_df = prepared.returns_df
        seed = prepared.seed

        if command == "run":
            return run_analysis_command(
                args,
                cfg_path,
                cfg,
                returns_path,
                returns_df,
                error=TrendCLIError,
                set_cache=set_cache_enabled,
                get_spec_preset=get_trend_spec_preset,
                list_spec_presets=list_trend_spec_presets,
                apply_spec_preset=_apply_trend_spec_preset,
                get_portfolio_preset=get_trend_preset,
                list_portfolio_presets=list_preset_slugs,
                apply_portfolio_preset=apply_trend_preset,
                load_universe=load_universe,
                apply_universe_mask=_apply_universe_mask,
                attach_universe_paths=_attach_universe_paths,
                find_existing_run=find_prior_run,
                calculate_run_id=working_run_id,
                get_default_log_path=run_logging.get_default_log_path,
                init_run_logger=run_logging.init_run_logger,
                log_step=maybe_log_step,
                run_pipeline=_run_pipeline,
                write_artifacts=_write_trend_run_artifacts,
                print_summary=_print_summary,
                finish_structured_log=_finish_structured_log,
                finalize_coverage=_finalize_config_coverage,
            )

        if command == "report":
            return run_report_command(
                args,
                cfg,
                returns_path,
                returns_df,
                error=TrendCLIError,
                default_formats=DEFAULT_REPORT_FORMATS,
                prepare_export_config=_prepare_export_config,
                run_pipeline=_run_pipeline,
                print_summary=_print_summary,
                write_report_files=_write_report_files,
                resolve_report_output_path=_resolve_report_output_path,
                generate_report=generate_unified_report,
                finalize_coverage=_finalize_config_coverage,
            )

        if command == "stress":
            if not args.scenario:
                raise TrendCLIError("The --scenario option is required for the 'stress' command")
            _adjust_for_scenario(cfg, args.scenario)
            export_dir = Path(args.out) if args.out else None
            _prepare_export_config(cfg, export_dir, None)
            run_pipeline = _run_pipeline
            result, run_id, _ = run_pipeline(
                cfg,
                returns_df,
                source_path=returns_path,
                log_file=None,
                structured_log=False,
                bundle=None,
            )
            print(f"Stress scenario '{args.scenario}' completed (seed={seed}).")
            print_summary = _print_summary
            print_summary(cfg, result)
            if export_dir:
                write_report = _write_report_files
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


def _noop_maybe_log_step(
    enabled: bool, run_id: str, event: str, message: str, **fields: Any
) -> None:
    return None


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
