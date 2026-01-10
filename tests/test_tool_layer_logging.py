from __future__ import annotations

import json
from pathlib import Path


from trend_analysis.tool_layer import _REDACTED_VALUE, ToolLayer


def test_tool_layer_logs_json_with_redaction(tmp_path: Path) -> None:
    log_path = tmp_path / "tool_calls.jsonl"
    tool = ToolLayer(log_path=log_path)
    config = {"analysis": {"top_n": 10}, "api_token": "secret-value"}
    patch = {
        "operations": [{"op": "set", "path": "analysis.top_n", "value": 12}],
        "summary": "Update top_n selection",
    }

    result = tool.apply_patch(config, patch)

    assert result.status == "success"
    payload = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert isinstance(payload["timestamp"], float)
    assert isinstance(payload["request_id"], str) and payload["request_id"]
    assert payload["tool"] == "apply_patch"
    assert isinstance(payload["parameters"], dict)
    assert isinstance(payload["output_summary"], str)
    assert payload["parameters"]["config"]["api_token"] == _REDACTED_VALUE


def test_tool_layer_rate_limit_blocks_excess_calls(tmp_path: Path) -> None:
    log_path = tmp_path / "tool_calls.jsonl"
    tool = ToolLayer(log_path=log_path, rate_limits={"apply_patch": 1})
    config = {"analysis": {"top_n": 10}}
    patch = {
        "operations": [{"op": "set", "path": "analysis.top_n", "value": 12}],
        "summary": "Update top_n selection",
    }

    first = tool.apply_patch(config, patch)
    second = tool.apply_patch(config, patch)

    assert first.status == "success"
    assert second.status == "error"
    assert "rate limit" in (second.message or "")
