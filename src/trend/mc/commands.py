"""Canonical ``trend mc`` parser and command handlers.

The scenario implementation remains shared with the compatibility CLI during
the migration.  Keeping the parser and dispatch boundary here lets callers use
``trend mc`` now without creating a third Monte Carlo implementation.
"""

from __future__ import annotations

import argparse
import json
import numbers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from trend.mc.viz import TrendCLIError
from trend_analysis.config import format_validation_messages, validate_config
from trend_analysis.io.market_data import MarketDataValidationError
from trend_analysis.monte_carlo import (
    MonteCarloRunner,
    MonteCarloScenario,
    MonteCarloSettings,
    list_scenarios,
    load_scenario,
)
from trend_analysis.monte_carlo.history import load_price_history
from trend_analysis.monte_carlo.registry import load_scenario_from_path
from trend_analysis.monte_carlo.results import MonteCarloResults, export_results


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
    """Dispatch canonical Monte Carlo commands without the legacy CLI."""
    command = getattr(args, "mc_command", None)
    if command == "viz":
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
    registry = (
        Path(args.registry).expanduser().resolve() if getattr(args, "registry", None) else None
    )
    if command == "list":
        try:
            entries = list_scenarios(
                tags=_parse_tags(getattr(args, "tags", None)), registry_path=registry
            )
        except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
            print(f"Failed to list Monte Carlo scenarios: {exc}", file=sys.stderr)
            return 1
        if getattr(args, "format", "table") == "json":
            print(
                json.dumps(
                    [
                        {
                            "name": item.name,
                            "description": item.description,
                            "tags": list(item.tags),
                            "path": str(item.path),
                        }
                        for item in entries
                    ],
                    indent=2,
                )
            )
        else:
            print(_render_table(entries))
        return 0
    if command not in {"validate", "run"}:
        print("Unknown Monte Carlo command.", file=sys.stderr)
        return 2
    try:
        scenario = _load_scenario(getattr(args, "scenario", "") or "", registry)
    except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
        print(f"Scenario {command} failed: {exc}", file=sys.stderr)
        return 1
    formats: list[str] = []
    if command == "run":
        try:
            _apply_overrides(scenario, args)
            formats = _parse_formats(getattr(args, "formats", None)) or ["csv"]
        except ValueError as exc:
            print(f"Scenario run failed: {exc}", file=sys.stderr)
            return 1
    errors = validate_mc_scenario(scenario)
    if errors:
        print(f"Scenario '{scenario.name}' failed validation:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    if command == "validate" or getattr(args, "dry_run", False):
        print(
            f"Scenario '{scenario.name}': OK"
            if command == "validate"
            else f"Scenario '{scenario.name}' validated. Dry run complete."
        )
        return 0
    try:
        price_history = load_price_history(Path(args.data)) if getattr(args, "data", None) else None
        settings = _resolved_mc_settings(scenario)
        progress_callback, close_progress = _build_progress_callback(
            total=int(settings.n_paths) if settings.n_paths is not None else 0,
            enabled=not getattr(args, "no_progress", False),
        )
        try:
            runner = MonteCarloRunner(scenario, base_config=None, price_history=price_history)
            results = runner.run(
                progress_callback=progress_callback,
                jobs=getattr(args, "jobs", None),
            )
        finally:
            close_progress()
        output_dir = Path(
            getattr(args, "out", None)
            or f"outputs/monte_carlo/{scenario.name}/{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        )
        exported = export_results(results, output_dir, formats=formats)
        write_mc_manifest(
            output_dir,
            scenario=scenario,
            results=results,
            overrides={
                key: getattr(args, key)
                for key in ("n_paths", "jobs", "seed")
                if getattr(args, key, None) is not None
            },
            exported_files=exported,
            data_path=Path(args.data) if getattr(args, "data", None) else None,
            jobs_used=runner._resolve_jobs(getattr(args, "jobs", None)),
        )
    except (MarketDataValidationError, OSError, ValueError) as exc:
        print(f"Scenario run failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Scenario run failed: {exc}", file=sys.stderr)
        return 2
    print(f"Monte Carlo run completed. Output: {output_dir}")
    return 0


def _parse_tags(raw: Sequence[str] | None) -> list[str]:
    return [tag.strip() for item in raw or [] for tag in str(item).split(",") if tag.strip()]


def _parse_formats(raw: Sequence[str] | None) -> list[str]:
    formats = [
        item.strip().lower()
        for value in raw or []
        for item in str(value).split(",")
        if item.strip()
    ]
    invalid = sorted(set(formats) - {"csv", "json", "parquet"})
    if invalid:
        raise ValueError(f"format overrides contains unsupported values: {', '.join(invalid)}")
    return formats


def _render_table(entries: Sequence[Any]) -> str:
    if not entries:
        return "No Monte Carlo scenarios found."
    columns = ("Name", "Tags", "Description", "Path")
    rows = [
        {
            "Name": entry.name,
            "Tags": ", ".join(
                sorted(dict.fromkeys(tag.strip() for tag in entry.tags if tag.strip()))
            )
            or "-",
            "Description": entry.description or "",
            "Path": str(entry.path),
        }
        for entry in entries
    ]
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows)) for column in columns
    }
    lines = [
        "  ".join(column.ljust(widths[column]) for column in columns),
        "  ".join("-" * widths[column] for column in columns),
    ]
    lines.extend(
        "  ".join(str(row[column]).ljust(widths[column]) for column in columns) for row in rows
    )
    return "\n".join(lines)


def _is_valid_progress_instance(candidate: Any) -> bool:
    return (
        candidate is not None
        and hasattr(candidate, "total")
        and all(
            callable(getattr(candidate, method, None)) for method in ("update", "refresh", "close")
        )
    )


def _build_progress_callback(
    *, total: int, enabled: bool
) -> tuple[Callable[[Mapping[str, Any]], None] | None, Callable[[], None]]:
    if not enabled:
        return None, lambda: None
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    if _is_valid_progress_instance(tqdm):
        bar = tqdm
        bar.total = total
        if hasattr(bar, "unit"):
            bar.unit = "path"
        if hasattr(bar, "file"):
            bar.file = sys.stderr
    elif callable(tqdm):
        try:
            bar = tqdm(total=total, unit="path", file=sys.stderr)
        except Exception:
            bar = None
    else:
        bar = None

    if not _is_valid_progress_instance(bar):
        state = {"last": -1}

        def text_callback(payload: Mapping[str, Any]) -> None:
            completed = int(payload.get("completed", 0))
            if completed == state["last"]:
                return
            state["last"] = completed
            print(f"Progress: {completed}/{int(payload.get('total', total))}", file=sys.stderr)

        return text_callback, lambda: None

    state = {"completed": 0}

    def callback(payload: Mapping[str, Any]) -> None:
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

    return callback, bar.close


def _load_scenario(raw: str, registry: Path | None) -> MonteCarloScenario:
    if not raw:
        raise ValueError("Scenario name is required")
    path = Path(raw).expanduser()
    if path.exists():
        return load_scenario_from_path(path)
    if path.suffix.lower() in {".yml", ".yaml"}:
        raise FileNotFoundError(f"Scenario config '{path}' does not exist")
    return load_scenario(raw, registry_path=registry)


def _resolved_mc_settings(scenario: MonteCarloScenario) -> MonteCarloSettings:
    settings = scenario.monte_carlo
    if not isinstance(settings, MonteCarloSettings):
        raise ValueError("monte_carlo settings are not resolved")
    return settings


def _apply_overrides(scenario: MonteCarloScenario, args: argparse.Namespace) -> None:
    settings = _resolved_mc_settings(scenario)
    scenario.monte_carlo = MonteCarloSettings(
        mode=settings.mode,
        n_paths=getattr(args, "n_paths", None) or settings.n_paths,
        horizon_years=settings.horizon_years,
        frequency=settings.frequency,
        seed=(
            getattr(args, "seed", None)
            if getattr(args, "seed", None) is not None
            else settings.seed
        ),
        jobs=(
            getattr(args, "jobs", None)
            if getattr(args, "jobs", None) is not None
            else settings.jobs
        ),
    )


def validate_mc_scenario(scenario: MonteCarloScenario) -> list[str]:
    try:
        runner = MonteCarloRunner(scenario)
    except Exception as exc:
        return [f"base_config: {exc}"]
    result = validate_config(
        dict(runner.base_config),
        base_path=Path(str(scenario.base_config)).parent,
        skip_required_fields=True,
    )
    schema_errors = [
        issue for issue in result.errors if issue.message.startswith("Unexpected field")
    ]
    if schema_errors:
        return [
            f"base_config: {message}"
            for message in format_validation_messages(
                result.model_copy(update={"errors": schema_errors}), include_warnings=False
            )
        ]
    return []


def write_mc_manifest(
    output_dir: Path,
    *,
    scenario: MonteCarloScenario,
    results: MonteCarloResults,
    overrides: Mapping[str, Any],
    exported_files: Mapping[str, Path],
    data_path: Path | None,
    jobs_used: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    mc_settings = _resolved_mc_settings(scenario)
    manifest_path.write_text(
        json.dumps(
            {
                "scenario": scenario.name,
                "description": scenario.description,
                "version": scenario.version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "base_config": str(scenario.base_config),
                "data_path": str(data_path) if data_path else None,
                "settings": {
                    "mode": mc_settings.mode,
                    "n_paths": mc_settings.n_paths,
                    "horizon_years": mc_settings.horizon_years,
                    "frequency": mc_settings.frequency,
                    "seed": mc_settings.seed,
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def is_valid_tqdm_instance(candidate: Any) -> bool:
    return (
        candidate is not None
        and all(callable(getattr(candidate, attr, None)) for attr in ("update", "refresh", "close"))
        and (
            getattr(candidate, "total", None) is None
            or (
                isinstance(getattr(candidate, "total"), numbers.Real)
                and float(getattr(candidate, "total")) >= 0
            )
        )
    )
