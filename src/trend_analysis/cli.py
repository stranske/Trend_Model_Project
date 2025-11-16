import argparse
import logging
import numbers
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from . import export, pipeline
from . import logging as run_logging
from .api import run_simulation
from .config import load_config
from .constants import DEFAULT_OUTPUT_DIRECTORY, DEFAULT_OUTPUT_FORMATS
from .data import load_csv
from .io.market_data import MarketDataValidationError
from .perf.rolling_cache import set_cache_enabled
from .presets import apply_trend_preset, get_trend_preset, list_preset_slugs
from .signal_presets import (
    TrendSpecPreset,
    get_trend_spec_preset,
    list_trend_spec_presets,
)
from .logging_setup import setup_logging

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


def _log_step(
    run_id: str, event: str, message: str, level: str = "INFO", **fields: Any
) -> None:
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


def maybe_log_step(
    enabled: bool, run_id: str, event: str, message: str, **fields: Any
) -> None:
    """Log a structured step when ``enabled`` is True."""
    if enabled:
        _log_step(run_id, event, message, **fields)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``trend-model`` command."""

    parser = argparse.ArgumentParser(prog="trend-model")
    parser.add_argument(
        "--check", action="store_true", help="Print environment info and exit"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("gui", help="Launch Streamlit interface")

    run_p = sub.add_parser("run", help="Run analysis pipeline")
    run_p.add_argument("-c", "--config", required=True, help="Path to YAML config")
    run_p.add_argument("-i", "--input", required=True, help="Path to returns CSV")
    run_p.add_argument(
        "--seed", type=int, help="Override random seed (takes precedence)"
    )
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
        cfg = load_config(args.config)
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
        try:
            loaded = load_market_data_csv(args.input)
        except MarketDataValidationError as exc:
            print(exc.user_message, file=sys.stderr)
            return 1
        if loaded is None:  # pragma: no cover - defensive fallback
            print("Failed to load data", file=sys.stderr)
            return 1
        df = loaded.frame if hasattr(loaded, "frame") else loaded
        split = cfg.sample_split
        required_keys = {"in_start", "in_end", "out_start", "out_end"}
        import uuid

        run_id = getattr(cfg, "run_id", None) or uuid.uuid4().hex[:12]
        try:
            setattr(cfg, "run_id", run_id)
        except Exception:
            # Some config implementations may forbid new attrs; proceed without persisting
            pass
        log_path = (
            Path(args.log_file)
            if args.log_file
            else run_logging.get_default_log_path(run_id)
        )
        do_structured = not args.no_structured_log
        if do_structured:
            run_logging.init_run_logger(run_id, log_path)
        maybe_log_step(
            do_structured,
            run_id,
            "start",
            "CLI run initialised",
            config_path=args.config,
        )
        if required_keys.issubset(split):
            maybe_log_step(
                do_structured,
                run_id,
                "load_data",
                "Loaded returns dataframe",
                rows=len(df),
            )
            run_result = run_simulation(cfg, df)
            maybe_log_step(
                do_structured,
                run_id,
                "pipeline_complete",
                "Pipeline execution finished",
                metrics_rows=len(run_result.metrics),
            )
            metrics_df = run_result.metrics
            res = run_result.details
            run_seed = run_result.seed
            # Attach time series required by export_bundle if present
            if isinstance(res, dict):
                # portfolio returns preference: user_weight then equal_weight fallback
                port_ser = None
                try:
                    port_ser = (
                        res.get("portfolio_user_weight")
                        or res.get("portfolio_equal_weight")
                        or res.get("portfolio_equal_weight_combined")
                    )
                except Exception:
                    port_ser = None
                if port_ser is not None:
                    setattr(run_result, "portfolio", port_ser)
                bench_map = res.get("benchmarks") if isinstance(res, dict) else None
                if isinstance(bench_map, dict) and bench_map:
                    # Pick first benchmark for manifest (simple case)
                    first_bench = next(iter(bench_map.values()))
                    setattr(run_result, "benchmark", first_bench)
                weights_user = (
                    res.get("weights_user_weight") if isinstance(res, dict) else None
                )
                if weights_user is not None:
                    setattr(run_result, "weights", weights_user)
        else:  # pragma: no cover - legacy fallback
            metrics_df = pipeline.run(cfg)
            res = pipeline.run_full(cfg)
            run_seed = getattr(cfg, "seed", 42)
        if not res:
            print("No results")
            return 0

        split = cfg.sample_split
        text = export.format_summary_text(
            res,
            str(split.get("in_start")),
            str(split.get("in_end")),
            str(split.get("out_start")),
            str(split.get("out_end")),
        )
        print(text)
        maybe_log_step(do_structured, run_id, "summary_render", "Printed summary text")

        cache_stats = _extract_cache_stats(res)
        if cache_stats:
            print("\nCache statistics:")
            print(f"  Entries: {cache_stats['entries']}")
            print(f"  Hits: {cache_stats['hits']}")
            print(f"  Misses: {cache_stats['misses']}")
            print(f"  Incremental updates: {cache_stats['incremental_updates']}")
            maybe_log_step(
                do_structured,
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
        if out_dir and out_formats:
            data = {"metrics": metrics_df}
            if any(f.lower() in {"excel", "xlsx"} for f in out_formats):
                sheet_formatter = export.make_summary_formatter(
                    res,
                    str(split.get("in_start")),
                    str(split.get("in_end")),
                    str(split.get("out_start")),
                    str(split.get("out_end")),
                )
                data["summary"] = pd.DataFrame()
                maybe_log_step(
                    do_structured,
                    run_id,
                    "export_start",
                    "Beginning export",
                    formats=out_formats,
                )
                export.export_to_excel(
                    data,
                    str(Path(out_dir) / f"{filename}.xlsx"),
                    default_sheet_formatter=sheet_formatter,
                )
                other = [f for f in out_formats if f.lower() not in {"excel", "xlsx"}]
                if other:
                    export.export_data(
                        data, str(Path(out_dir) / filename), formats=other
                    )
                else:
                    export.export_data(
                        data, str(Path(out_dir) / filename), formats=out_formats
                    )
            maybe_log_step(
                do_structured,
                run_id,
                "export_complete",
                "Export finished",
            )

        # Optional bundle export (reproducibility manifest + hashes)
        if args.bundle:
            from .api import RunResult as _RR
            from .export.bundle import export_bundle

            bundle_path = Path(args.bundle)
            if bundle_path.is_dir():
                bundle_path = bundle_path / "analysis_bundle.zip"
            # Build a minimal RunResult-like shim if we executed legacy path
            rr = locals().get("run_result")
            if rr is None:  # legacy path
                env = {
                    "python": sys.version.split()[0],
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                }
                rr = _RR(metrics_df, res, run_seed, env)
            # Attach config + seed for export_bundle
            setattr(rr, "config", getattr(cfg, "__dict__", {}))
            setattr(rr, "input_path", Path(args.input))
            export_bundle(rr, bundle_path)
            print(f"Bundle written: {bundle_path}")
            maybe_log_step(
                do_structured,
                run_id,
                "bundle_complete",
                "Reproducibility bundle written",
                bundle=str(bundle_path),
            )
        maybe_log_step(
            do_structured,
            run_id,
            "end",
            "CLI run complete",
            log_file=str(log_path),
        )
        return 0

    # This shouldn't be reached with required=True.
    return 0


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
