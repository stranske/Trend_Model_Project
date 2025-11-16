from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Protocol, cast

import numpy as np
import pandas as pd

from trend.reporting import generate_unified_report
from trend.reporting.quick_summary import main as quick_summary_main
from trend_analysis import export
from trend_analysis import logging as run_logging
from trend_analysis.api import RunResult, run_simulation
from trend_analysis.config import load as load_config
from trend_analysis.constants import DEFAULT_OUTPUT_DIRECTORY, DEFAULT_OUTPUT_FORMATS
from trend_analysis.data import load_csv
from trend_model.spec import ensure_run_spec

LegacyExtractCacheStats = Callable[[object], dict[str, int] | None]


class LegacyMaybeLogStep(Protocol):
    def __call__(
        self, enabled: bool, run_id: str, event: str, message: str, **fields: Any
    ) -> None:
        # Protocol method intentionally empty; implementors provide behaviour.
        ...


def _noop_maybe_log_step(
    enabled: bool, run_id: str, event: str, message: str, **fields: Any
) -> None:
    return None


_legacy_cli_module: ModuleType | None = None
_legacy_extract_cache_stats: LegacyExtractCacheStats | None = None
_legacy_maybe_log_step: LegacyMaybeLogStep = _noop_maybe_log_step


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

    return module or _legacy_cli_module


_refresh_legacy_cli_module()


APP_PATH = Path(__file__).resolve().parents[2] / "streamlit_app" / "app.py"

DEFAULT_REPORT_FORMATS = ("csv", "json", "xlsx", "txt")

SCENARIO_WINDOWS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "2008": (("2006-01", "2007-12"), ("2008-01", "2009-12")),
    "2020": (("2018-01", "2019-12"), ("2020-01", "2021-12")),
}


def _legacy_callable(name: str, fallback: Callable[..., Any]) -> Callable[..., Any]:
    module = _refresh_legacy_cli_module()
    if module is not None:
        attr = getattr(module, name, None)
        if callable(attr):
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

    report_p = sub.add_parser(
        "report", help="Generate summary artefacts for a configuration"
    )
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

    stress_p = sub.add_parser(
        "stress", help="Run the pipeline against a canned stress scenario"
    )
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

    sub.add_parser("app", help="Launch the Streamlit application")

    quick_p = sub.add_parser(
        "quick-report", help="Build a compact HTML report from run artefacts"
    )
    quick_p.add_argument(
        "--run-id", help="Run identifier (defaults to artefact inference)"
    )
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

    return parser


def _resolve_returns_path(config_path: Path, cfg: Any, override: str | None) -> Path:
    if override:
        path = Path(override)
    else:
        csv_path = cfg.data.get("csv_path") if hasattr(cfg, "data") else None
        if not csv_path:
            msg = "Configuration must define data.csv_path or use --returns"
            raise TrendCLIError(msg)
        path = Path(csv_path)
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _ensure_dataframe(path: Path) -> pd.DataFrame:
    try:
        df = load_csv(str(path), errors="raise")
    except TypeError:
        df = load_csv(str(path))
    if df is None:
        raise FileNotFoundError(str(path))
    return df


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


def _prepare_export_config(
    cfg: Any, directory: Path | None, formats: Iterable[str] | None
) -> None:
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
    run_id = getattr(cfg, "run_id", None) or uuid.uuid4().hex[:12]
    try:
        setattr(cfg, "run_id", run_id)
    except Exception:
        pass

    log_path = None
    if structured_log:
        log_path = log_file or run_logging.get_default_log_path(run_id)
        run_logging.init_run_logger(run_id, log_path)
    _legacy_maybe_log_step(
        structured_log, run_id, "start", "trend CLI execution started"
    )

    result = run_simulation(cfg, returns_df)
    details = result.details
    if isinstance(details, dict):
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
    _persist_turnover_ledger(run_id, getattr(result, "details", {}))

    if bundle:
        _write_bundle(cfg, result, source_path, Path(bundle), structured_log, run_id)

    return result, run_id, log_path


def _handle_exports(
    cfg: Any, result: RunResult, structured_log: bool, run_id: str
) -> None:
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


def _write_report_files(
    out_dir: Path, cfg: Any, result: RunResult, *, run_id: str
) -> None:
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
    _maybe_write_turnover_csv(out_dir, getattr(result, "details", {}))
    print(f"Report artefacts written to {out_dir}")


def _resolve_report_output_path(
    output: str | None, export_dir: Path | None, run_id: str
) -> Path:
    if output:
        base = Path(output).expanduser()
        if base.exists() and base.is_dir():
            return base / f"trend_report_{run_id}.html"
        if base.suffix.lower() in {".html", ".htm"}:
            return base
        if base.suffix:
            return base
        return base / f"trend_report_{run_id}.html"
    base_dir = export_dir if export_dir is not None else Path.cwd()
    return base_dir / f"trend_report_{run_id}.html"


def _json_default(obj: Any) -> Any:  # pragma: no cover - helper
    if isinstance(obj, pd.Series):
        data: dict[str, Any] = {}
        for key, value in obj.items():
            # JSON objects require string keys, so coerce anything non-string
            coerced_key: str
            if isinstance(key, str):
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


def _maybe_write_turnover_csv(directory: Path, details: Any) -> Path | None:
    if not isinstance(details, Mapping):
        return None
    diag = details.get("risk_diagnostics")
    if not isinstance(diag, Mapping):
        return None
    turnover_obj = diag.get("turnover")
    if isinstance(turnover_obj, pd.Series):
        series = turnover_obj.copy()
    elif isinstance(turnover_obj, Mapping):
        series = pd.Series(turnover_obj)
    elif isinstance(turnover_obj, (list, tuple)):
        series = pd.Series(turnover_obj)
    else:
        return None
    try:
        series = series.astype(float)
    except (TypeError, ValueError):
        return None
    if series.empty:
        return None
    series = series.sort_index()
    frame = series.rename("turnover").to_frame()
    frame.index.name = "Date"
    path = directory / "turnover.csv"
    frame.to_csv(path)
    return path


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
    if "transaction_cost_bps" not in portfolio:
        raise TrendCLIError(
            "Configuration must define portfolio.transaction_cost_bps for honest costs."
        )
    try:
        cost = float(portfolio["transaction_cost_bps"])
    except (TypeError, ValueError) as exc:
        raise TrendCLIError("portfolio.transaction_cost_bps must be numeric") from exc
    if cost < 0:
        raise TrendCLIError("portfolio.transaction_cost_bps cannot be negative")


def _persist_turnover_ledger(run_id: str, details: Any) -> Path | None:
    if not isinstance(details, Mapping):
        return None
    diag = details.get("risk_diagnostics")
    if not isinstance(diag, Mapping):
        return None
    turnover_obj = diag.get("turnover")
    if turnover_obj is None:
        return None
    if isinstance(turnover_obj, pd.Series):
        if turnover_obj.empty:
            return None
    elif isinstance(turnover_obj, Mapping):
        if not turnover_obj:
            return None
    elif isinstance(turnover_obj, (list, tuple)):
        if not turnover_obj:
            return None
    else:
        return None
    target_dir = Path("perf") / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    path = _maybe_write_turnover_csv(target_dir, details)
    if path is not None:
        print(f"Turnover ledger written to {path}")
    return path


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
    cfg = load_config(cfg_path)
    ensure_run_spec(cfg, base_path=cfg_path.parent)
    return cfg_path, cfg


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)

        command = args.subcommand

        if command == "app":
            proc = subprocess.run(["streamlit", "run", str(APP_PATH)])
            return proc.returncode

        if command == "quick-report":
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

        if command not in {"run", "report", "stress"}:
            raise TrendCLIError(f"Unknown command: {command}")

        if not args.config:
            raise TrendCLIError(
                f"The --config option is required for the '{command}' command"
            )

        load_config_fn = _legacy_callable("_load_configuration", _load_configuration)
        cfg_path, cfg = load_config_fn(args.config)
        ensure_run_spec(cfg, base_path=cfg_path.parent)
        resolve_returns = _legacy_callable(
            "_resolve_returns_path", _resolve_returns_path
        )
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
            return 0

        if command == "report":
            export_dir = Path(args.out).resolve() if args.out else None
            if export_dir is None and not args.output:
                raise TrendCLIError(
                    "The 'report' command requires --out for artefacts or --output for the HTML report"
                )
            formats = args.formats or DEFAULT_REPORT_FORMATS
            _prepare_export_config(
                cfg, export_dir, formats if export_dir is not None else None
            )
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
                write_report = _legacy_callable(
                    "_write_report_files", _write_report_files
                )
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
            return 0

        if command == "stress":
            if not args.scenario:
                raise TrendCLIError(
                    "The --scenario option is required for the 'stress' command"
                )
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
                write_report = _legacy_callable(
                    "_write_report_files", _write_report_files
                )
                write_report(export_dir, cfg, result, run_id=run_id)
            return 0

        raise TrendCLIError(f"Unknown command: {command}")
    except TrendCLIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
