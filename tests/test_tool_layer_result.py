from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from trend_analysis.tool_layer import ToolLayer, ToolResult


def test_tool_layer_returns_toolresult_success(tmp_path: Path) -> None:
    log_path = tmp_path / "tool_calls.jsonl"
    tool = ToolLayer(log_path=log_path)
    config = {"analysis": {"top_n": 10}}
    patch = {
        "operations": [{"op": "set", "path": "analysis.top_n", "value": 12}],
        "summary": "Update top_n selection",
    }

    result = tool.preview_diff(config, patch)

    assert isinstance(result, ToolResult)
    assert isinstance(result, BaseModel)
    assert result.status == "success"
    assert result.message is None
    assert isinstance(result.data, str)
    assert result.elapsed_ms >= 0


def test_tool_layer_returns_toolresult_error(tmp_path: Path) -> None:
    log_path = tmp_path / "tool_calls.jsonl"
    tool = ToolLayer(log_path=log_path)
    patch = {
        "operations": [{"op": "set", "path": "analysis.top_n", "value": 12}],
        "summary": "Update top_n selection",
    }

    result = tool.apply_patch(["not", "a", "mapping"], patch)

    assert isinstance(result, ToolResult)
    assert result.status == "error"
    assert isinstance(result.message, str) and result.message
    assert result.data is None
    assert result.elapsed_ms >= 0
