from __future__ import annotations

import ast
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("field", ("per_trade_bps", "half_spread_bps"))
@pytest.mark.parametrize("value", (0.5, 1.4, 2.5, 9.9, float("nan"), float("inf")))
def test_run_path_preserves_finite_cost_bps_and_rejects_non_finite_values(
    monkeypatch, field, value
):
    """Both run mappings retain finite bps and default unsafe values independently."""

    streamlit_stub = SimpleNamespace(session_state={})
    streamlit_stub.cache_data = lambda *args, **kwargs: (
        args[0] if args and callable(args[0]) else (lambda fn: fn)
    )
    streamlit_stub.cache_resource = streamlit_stub.cache_data
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)

    from streamlit_app.components.analysis_runner import (
        _build_portfolio_config,
    )
    from trend_analysis.config.ui_mapping import build_portfolio_config

    model_state = {field: value}
    expected = value if math.isfinite(value) and value >= 0 else 0.0

    assert _build_portfolio_config(model_state, {})["cost_model"][field] == expected
    assert build_portfolio_config(model_state, {})["cost_model"][field] == expected


@pytest.mark.parametrize(
    ("label", "maximum"),
    (("Transaction Cost (bps)", 100.0), ("Slippage (bps)", 50.0)),
)
def test_model_form_uses_fractional_safe_cost_inputs(label, maximum):
    """The visible Streamlit form must not truncate the fractional cost model."""

    page = Path(__file__).parents[1] / "streamlit_app/pages/2_Model.py"
    tree = ast.parse(page.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "number_input"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == label
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert keywords["min_value"].value == 0.0
    assert keywords["max_value"].value == maximum
    assert keywords["step"].value == 0.1
    assert isinstance(keywords["value"], ast.Call)
    assert keywords["value"].func.id == "_finite_nonnegative_float"
