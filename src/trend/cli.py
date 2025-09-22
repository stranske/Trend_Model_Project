from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from trend_analysis import export
from trend_analysis import logging as run_logging
from trend_analysis.api import RunResult, run_simulation
from trend_analysis.config import load as load_config
from trend_analysis.constants import DEFAULT_OUTPUT_DIRECTORY, DEFAULT_OUTPUT_FORMATS
from trend_analysis.data import load_csv

try:  # ``trend_analysis.cli`` is heavy but provides useful helpers
    from trend_analysis.cli import (
        _extract_cache_stats as _legacy_extract_cache_stats,
        maybe_log_step as _legacy_maybe_log_step,
    )
except Exception:  # pragma: no cover - defensive fallback
    _legacy_extract_cache_stats = None  # type: ignore[assignment]

    def _legacy_maybe_log_step(
        enabled: bool, run_id: str, event: str, message: str, **fields: Any
    ) -> None:  # noqa: D401 - simple noop
        """Fallback when legacy helpers unavailable (signature matches maybe_log_step)."""
        return None


APP_PATH = Path(__file__).resolve().parents[2] / "streamlit_app" / "app.py"

DEFAULT_REPORT_FORMATS = ("csv", "json", "xlsx", "txt")

SCENARIO_WINDOWS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "2008": (("2006-01", "2007-12"), ("2008-01", "2009-12")),
    "2020": (("2018-01", "2019-12"), ("2020-01", "2021-12")),
}


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
        "--formats",
        nargs="+",
        choices=DEFAULT_REPORT_FORMATS,
        help="Subset of export formats (default: csv json xlsx txt)",
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
    df = load_csv(str(path))
    if df is None:
        raise FileNotFoundError(path)
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
    print(f"Report artefacts written to {out_dir}")


def _json_default(obj: Any) -> Any:  # pragma: no cover - helper
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


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
    return cfg_path, load_config(cfg_path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)

        command = args.subcommand

        if command == "app":
            proc = subprocess.run(["streamlit", "run", str(APP_PATH)])
            return proc.returncode

        if command in {"run", "report", "stress"} and not args.config:
            raise TrendCLIError(
                f"The --config option is required for the '{command}' command"
            )

        cfg_path, cfg = _load_configuration(args.config)
        returns_path = _resolve_returns_path(
            cfg_path, cfg, getattr(args, "returns", None)
        )
        returns_df = _ensure_dataframe(returns_path)
        seed = _determine_seed(cfg, getattr(args, "seed", None))

        if command == "run":
            result, run_id, log_path = _run_pipeline(
                cfg,
                returns_df,
                source_path=returns_path,
                log_file=Path(args.log_file) if args.log_file else None,
                structured_log=not args.no_structured_log,
                bundle=Path(args.bundle) if args.bundle else None,
            )
            _print_summary(cfg, result)
            if log_path:
                print(f"Structured log: {log_path}")
            return 0

        if command == "report":
            if not args.out:
                raise TrendCLIError(
                    "The --out option is required for the 'report' command"
                )
            formats = args.formats or DEFAULT_REPORT_FORMATS
            _prepare_export_config(cfg, Path(args.out), formats)
            result, run_id, _ = _run_pipeline(
                cfg,
                returns_df,
                source_path=returns_path,
                log_file=None,
                structured_log=False,
                bundle=None,
            )
            _print_summary(cfg, result)
            _write_report_files(Path(args.out), cfg, result, run_id=run_id)
            return 0

        if command == "stress":
            if not args.scenario:
                raise TrendCLIError(
                    "The --scenario option is required for the 'stress' command"
                )
            _adjust_for_scenario(cfg, args.scenario)
            export_dir = Path(args.out) if args.out else None
            _prepare_export_config(cfg, export_dir, None)
            result, run_id, _ = _run_pipeline(
                cfg,
                returns_df,
                source_path=returns_path,
                log_file=None,
                structured_log=False,
                bundle=None,
            )
            print(f"Stress scenario '{args.scenario}' completed (seed={seed}).")
            _print_summary(cfg, result)
            if export_dir:
                _write_report_files(export_dir, cfg, result, run_id=run_id)
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
