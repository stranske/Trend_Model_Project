from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path


ENGINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "trend_analysis"
    / "multi_period"
    / "engine.py"
)
MAX_RUN_BODY_LINES = 1_500
MAX_RUN_INDENT_DEPTH = 6


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
    source = ENGINE_PATH.read_text()
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    current_depth = 0
    max_depth = 0
    for token in tokens:
        line_no = token.start[0]
        if line_no <= node.lineno or line_no > node.end_lineno:
            continue
        if token.type == tokenize.INDENT:
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif token.type == tokenize.DEDENT:
            current_depth = max(0, current_depth - 1)
    return max_depth


def test_run_body_is_below_refactor_size_target() -> None:
    body_lines = _run_body_line_count()
    max_indent = _run_max_indent_depth()

    assert body_lines < MAX_RUN_BODY_LINES
    assert max_indent <= MAX_RUN_INDENT_DEPTH
