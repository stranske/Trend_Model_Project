"""Structural ownership contracts for the canonical CLI front door."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "src/trend/cli.py"
MC_VIZ_PATH = ROOT / "src/trend/mc/viz.py"

MIGRATED_MC_VIZ_HANDLERS = {
    "_validate_mc_viz_output_flags",
    "_read_mc_frame",
    "_load_mc_bundle_frames",
    "_parse_mc_chart_selection",
    "_mc_chart_builders",
    "_export_mc_chart_artifacts",
    "_inject_mc_html_chart_markers",
    "_run_mc_viz_command",
}


def _defined_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_cli_does_not_redefine_migrated_command_handlers() -> None:
    """The parser front door delegates MC visualization to its owning module."""

    cli_functions = _defined_functions(CLI_PATH)
    viz_functions = _defined_functions(MC_VIZ_PATH)

    assert "build_parser" in cli_functions
    assert "main" in cli_functions
    assert not (MIGRATED_MC_VIZ_HANDLERS & cli_functions)
    assert {
        "_read_mc_frame",
        "_load_mc_bundle_frames",
        "_parse_mc_chart_selection",
        "_mc_chart_builders",
        "_export_mc_chart_artifacts",
        "_inject_mc_html_chart_markers",
        "execute_mc_viz_cli",
    } <= viz_functions
