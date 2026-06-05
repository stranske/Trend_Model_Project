from __future__ import annotations

import ast
from pathlib import Path


ENGINE_PATH = Path(__file__).resolve().parents[1] / "src/trend_analysis/multi_period/engine.py"

BASELINE_RUN_BODY_LINES = 3620
TARGET_RUN_BODY_LINES = 2896
BASELINE_MAX_INDENT_DEPTH = 10
TARGET_MAX_INDENT_DEPTH = 9


def _run_function_node() -> ast.FunctionDef:
    module = ast.parse(ENGINE_PATH.read_text())
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return node
    raise AssertionError("run() function not found")


def _run_body_line_count() -> int:
    node = _run_function_node()
    assert node.end_lineno is not None
    return node.end_lineno - node.lineno + 1


def _run_max_indent_depth() -> int:
    node = _run_function_node()
    assert node.end_lineno is not None
    lines = ENGINE_PATH.read_text().splitlines()[node.lineno - 1 : node.end_lineno]
    base_indent = len(lines[0]) - len(lines[0].lstrip())
    max_depth = 0
    for line in lines[1:]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        max_depth = max(max_depth, (indent - base_indent) // 4)
    return max_depth


def test_run_body_line_count_and_indent_depth_meet_refactor_targets() -> None:
    body_lines = _run_body_line_count()
    max_indent = _run_max_indent_depth()

    assert body_lines <= TARGET_RUN_BODY_LINES
    assert body_lines <= int(BASELINE_RUN_BODY_LINES * 0.8)
    assert max_indent <= TARGET_MAX_INDENT_DEPTH
    assert max_indent <= BASELINE_MAX_INDENT_DEPTH - 1
