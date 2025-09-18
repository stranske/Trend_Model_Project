import argparse
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
import numpy as np

import numpy as np
import pandas as pd

from . import export, pipeline
from .api import run_simulation
from .config import load_config
from .constants import DEFAULT_OUTPUT_DIRECTORY, DEFAULT_OUTPUT_FORMATS
from .data import load_csv

APP_PATH = Path(__file__).resolve().parents[2] / "streamlit_app" / "app.py"
LOCK_PATH = Path(__file__).resolve().parents[2] / "requirements.lock"


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

    if args.check:
        return check_environment()

    if args.command == "gui":
        proc = subprocess.run(["streamlit", "run", str(APP_PATH)])
        return proc.returncode

    if args.command == "run":
        cfg = load_config(args.config)
        cli_seed = args.seed
        env_seed = os.getenv("TREND_SEED")
        # Precedence: CLI flag > TREND_SEED > config.seed > default 42
        if cli_seed is not None:
            cfg.seed = int(cli_seed)  # type: ignore[attr-defined]
        elif env_seed is not None and env_seed.isdigit():
            cfg.seed = int(env_seed)  # type: ignore[attr-defined]
        df = load_csv(args.input)
        assert df is not None  # narrow type for type-checkers
        split = cfg.sample_split
        required_keys = {"in_start", "in_end", "out_start", "out_end"}
        from .logging import (
            get_default_log_path,
            init_run_logger,
            log_step,
        )
        import uuid

        run_id = getattr(cfg, "run_id", None) or uuid.uuid4().hex[:12]
        try:
            setattr(cfg, "run_id", run_id)  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            # Some config implementations may forbid new attrs; proceed without persisting
            pass
        log_path = (
            Path(args.log_file) if args.log_file else get_default_log_path(run_id)
        )
        do_structured = not args.no_structured_log
        if do_structured:
            init_run_logger(run_id, log_path)
            log_step(run_id, "start", "CLI run initialised", config_path=args.config)
        if required_keys.issubset(split):
            if do_structured:
                log_step(run_id, "load_data", "Loaded returns dataframe", rows=len(df))
            run_result = run_simulation(cfg, df)
            if do_structured:
                log_step(
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
                    run_result.portfolio = port_ser  # type: ignore[attr-defined]
                bench_map = res.get("benchmarks") if isinstance(res, dict) else None
                if isinstance(bench_map, dict) and bench_map:
                    # Pick first benchmark for manifest (simple case)
                    first_bench = next(iter(bench_map.values()))
                    run_result.benchmark = first_bench  # type: ignore[attr-defined]
                weights_user = (
                    res.get("weights_user_weight") if isinstance(res, dict) else None
                )
                if weights_user is not None:
                    run_result.weights = weights_user  # type: ignore[attr-defined]
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
        if do_structured:
            log_step(run_id, "summary_render", "Printed summary text")

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
                if do_structured:
                    log_step(
                        run_id, "export_start", "Beginning export", formats=out_formats
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
            if do_structured:
                log_step(run_id, "export_complete", "Export finished")

        # Optional bundle export (reproducibility manifest + hashes)
        if args.bundle:
            from .export.bundle import export_bundle
            from .api import RunResult as _RR

            bundle_path = Path(args.bundle)
            if bundle_path.is_dir():
                bundle_path = bundle_path / "analysis_bundle.zip"
            # Build a minimal RunResult-like shim if we executed legacy path
            if "run_result" in locals():  # modern path
                rr = run_result
            else:
                env = {
                    "python": sys.version.split()[0],
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                }
                rr = _RR(metrics_df, res, run_seed, env)
            # Attach config + seed for export_bundle
            rr.config = getattr(cfg, "__dict__", {})  # type: ignore[attr-defined]
            rr.input_path = Path(args.input)  # type: ignore[attr-defined]
            export_bundle(rr, bundle_path)
            print(f"Bundle written: {bundle_path}")
            if do_structured:
                log_step(
                    run_id,
                    "bundle_complete",
                    "Reproducibility bundle written",
                    bundle=str(bundle_path),
                )
        if do_structured:
            log_step(run_id, "end", "CLI run complete", log_file=str(log_path))
        return 0

    # This shouldn't be reached with required=True.
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
