"""Canonical ``trend mc`` parser and command handlers.

The scenario implementation remains shared with the compatibility CLI during
the migration.  Keeping the parser and dispatch boundary here lets callers use
``trend mc`` now without creating a third Monte Carlo implementation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trend.mc.viz import TrendCLIError
from trend_analysis import cli as _legacy_cli


def add_mc_subparsers(parent: argparse.ArgumentParser) -> None:
    """Add the complete Monte Carlo command tree to the canonical parser."""

    sub = parent.add_subparsers(dest="mc_command", required=True)

    list_parser = sub.add_parser("list", help="List registered Monte Carlo scenarios")
    list_parser.add_argument("--tags", action="append", default=[], help="Filter by scenario tags")
    list_parser.add_argument("--format", choices=("table", "json"), default="table")
    list_parser.add_argument("--registry", help="Override the scenario registry path")

    validate_parser = sub.add_parser("validate", help="Validate Monte Carlo scenarios")
    validate_parser.add_argument("scenario", nargs="?", help="Scenario name or config path")
    validate_parser.add_argument(
        "--tags", action="append", default=[], help="Filter by scenario tags"
    )
    validate_parser.add_argument("--registry", help="Override the scenario registry path")

    run_parser = sub.add_parser("run", help="Run Monte Carlo scenarios")
    run_parser.add_argument("--scenario", required=True, help="Scenario name or config path")
    run_parser.add_argument("--data", help="CSV/Parquet price or returns history")
    run_parser.add_argument("--out", help="Output directory for the Monte Carlo bundle")
    run_parser.add_argument(
        "--formats", action="append", default=[], help="Repeatable CSV/JSON/Parquet formats"
    )
    run_parser.add_argument("--n-paths", type=int, help="Override number of Monte Carlo paths")
    run_parser.add_argument("--jobs", type=int, help="Override parallel job count")
    run_parser.add_argument("--seed", type=int, help="Override Monte Carlo seed")
    run_parser.add_argument("--dry-run", action="store_true", help="Validate without executing")
    run_parser.add_argument("--no-progress", action="store_true", help="Disable progress output")
    run_parser.add_argument("--registry", help="Override the scenario registry path")

    viz_parser = sub.add_parser("viz", help="Render Monte Carlo chart artifacts from a bundle")
    viz_parser.add_argument("--bundle", required=True, help="Monte Carlo bundle directory")
    viz_parser.add_argument("--out", type=Path, required=True, help="Artifact output directory")
    viz_parser.add_argument("--charts", default="fan,path_dist,risk_return")
    nav_paths_group = viz_parser.add_mutually_exclusive_group()
    nav_paths_group.add_argument("--fold", type=int, help="Fold-exported NAV paths to load")
    nav_paths_group.add_argument("--nav-paths", type=Path, dest="nav_paths")
    viz_parser.add_argument("--html", action="store_true")
    viz_parser.add_argument("--json", action="store_true")
    viz_parser.add_argument("--png", action="store_true")


def handle_mc_command(args: argparse.Namespace) -> int:
    """Run the shared scenario behavior behind the canonical parser."""

    if getattr(args, "mc_command", None) == "viz":
        from trend.mc.viz import execute_mc_viz_cli

        try:
            return execute_mc_viz_cli(
                bundle_path=args.bundle,
                out_dir=args.out,
                charts=args.charts,
                fold_id=getattr(args, "fold", None),
                nav_paths=getattr(args, "nav_paths", None),
                html=args.html,
                json=args.json,
                png=args.png,
            )
        except TrendCLIError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        except (OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
    return _legacy_cli._handle_mc_command(args)


# Transitional public helpers keep focused scenario tests independent of the
# compatibility entry point while the implementation is shared.
validate_mc_scenario = _legacy_cli._validate_mc_scenario
write_mc_manifest = _legacy_cli._write_mc_manifest
is_valid_tqdm_instance = _legacy_cli._is_valid_tqdm_instance
