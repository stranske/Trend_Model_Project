from __future__ import annotations

from trend_analysis.tool_layer import ToolLayer


def test_apply_patch_updates_config_mapping() -> None:
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

    updated = tool.apply_patch(config, patch)

    assert updated["analysis"]["top_n"] == 12
    assert config["analysis"]["top_n"] == 10
