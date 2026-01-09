from __future__ import annotations

from trend_analysis.tool_layer import ToolLayer


def test_preview_diff_returns_unified_diff() -> None:
    tool = ToolLayer()
    config = {"analysis": {"top_n": 10}}
    patch = {
        "operations": [
            {
                "op": "set",
                "path": "analysis.top_n",
                "value": 12,
            }
        ],
        "summary": "Update top_n selection",
    }

    diff = tool.preview_diff(config, patch)

    assert "--- before" in diff
    assert "+++ after" in diff
    assert "top_n: 10" in diff
    assert "top_n: 12" in diff
    assert config["analysis"]["top_n"] == 10
