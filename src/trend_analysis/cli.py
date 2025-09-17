import argparse
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pandas as pd

from . import export
from .api import RunResult, run_simulation
from .config import ConfigType, load_config
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


REQUIRED_SAMPLE_KEYS = {"in_start", "in_end", "out_start", "out_end"}


def run_cli_simulation(config: ConfigType, returns: pd.DataFrame) -> RunResult:
    """Execute the CLI pipeline via :func:`run_simulation`.

    Parameters
    ----------
    config : ConfigType
        Loaded configuration object.
    returns : pd.DataFrame
        Returns data including a ``Date`` column.

    Returns
    -------
    RunResult
        Structured output from :func:`trend_analysis.api.run_simulation`.
    """

    split = getattr(config, "sample_split", {})
    missing = [key for key in REQUIRED_SAMPLE_KEYS if not split.get(key)]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(
            f"Configuration sample_split is missing required keys: {joined}"
        )

    return run_simulation(config, returns)


def run_cli_simulation_from_paths(
    config_path: str, returns_path: str
) -> tuple[RunResult, ConfigType]:
    """Load configuration and returns from disk then run the pipeline."""

    config = load_config(config_path)
    returns = load_csv(returns_path)
    if returns is None:
        raise FileNotFoundError(returns_path)
    result = run_cli_simulation(config, returns)
    return result, config


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
        run_result, cfg = run_cli_simulation_from_paths(args.config, args.input)
        metrics_df = run_result.metrics
        res = run_result.details
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
        return 0

    # This shouldn't be reached with required=True.
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
