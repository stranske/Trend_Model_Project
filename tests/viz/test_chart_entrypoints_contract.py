"""Contract tests for chart entrypoint API consistency."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from trend_analysis.viz import sharpe_ladder
from trend_analysis.viz.charts import corr_heatmap, rolling_panel, seasonality_heatmap

ENTRYPOINTS = (
    corr_heatmap.build_figure,
    rolling_panel.build_figure,
    seasonality_heatmap.build_figure,
    sharpe_ladder.build_figure,
)


def test_all_chart_modules_expose_build_figure() -> None:
    for entrypoint in ENTRYPOINTS:
        assert callable(entrypoint)


def test_primary_input_parameter_name_is_consistent() -> None:
    first_param_names = [
        next(iter(inspect.signature(fn).parameters.values())).name for fn in ENTRYPOINTS
    ]
    assert first_param_names == ["data", "data", "data", "data"]


def _load_entrypoint_ast(entrypoint: object) -> ast.FunctionDef:
    source_path = inspect.getsourcefile(entrypoint)
    if source_path is None:
        raise AssertionError(f"Unable to resolve source file for {entrypoint!r}")

    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"), filename=source_path)
    function_name = getattr(entrypoint, "__name__", None)
    if not isinstance(function_name, str):
        raise AssertionError(f"Unable to resolve function name for {entrypoint!r}")

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"Unable to locate function definition for {function_name!r}")


def _find_streamlit_attribute_violations(func_def: ast.FunctionDef) -> list[int]:
    bad_line_numbers: list[int] = []
    for statement in func_def.body:
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in {"st", "streamlit"}
            ):
                bad_line_numbers.append(node.lineno)
    return bad_line_numbers


def _load_function_def_from_source(source: str, function_name: str = "build_figure") -> ast.FunctionDef:
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"Unable to locate function definition for {function_name!r}")


def test_public_entrypoints_do_not_call_streamlit_api() -> None:
    for entrypoint in ENTRYPOINTS:
        entrypoint_ast = _load_entrypoint_ast(entrypoint)
        violations = _find_streamlit_attribute_violations(entrypoint_ast)
        assert not violations, (
            f"{entrypoint.__module__}.{entrypoint.__name__} calls Streamlit in executable body "
            f"at lines {violations}"
        )


@pytest.mark.parametrize(
    ("source", "expected_violations"),
    [
        (
            """
def build_figure(data):
    st.write(data)
    return data
""",
            [3],
        ),
        (
            """
def build_figure(data):
    streamlit.write(data)
    return data
""",
            [3],
        ),
        (
            '''
def build_figure(data):
    # st.write(data)
    note = "streamlit.write(data)"
    return data
''',
            [],
        ),
        (
            """
@st.cache_data
def build_figure(data):
    return data
""",
            [],
        ),
        (
            """
HELPER = st.write

def build_figure(data):
    return data
""",
            [],
        ),
    ],
)
def test_streamlit_attribute_detection_cases(source: str, expected_violations: list[int]) -> None:
    function_def = _load_function_def_from_source(source)
    assert _find_streamlit_attribute_violations(function_def) == expected_violations
