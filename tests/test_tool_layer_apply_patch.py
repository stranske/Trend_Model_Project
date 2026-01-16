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

    result = tool.apply_patch(config, patch)

    assert result.status == "success"
    assert result.data["analysis"]["top_n"] == 12
    assert config["analysis"]["top_n"] == 10


def test_apply_patch_requires_confirmation_for_risky_patch() -> None:
    tool = ToolLayer()
    config = {"portfolio": {"constraints": {"max_weight": 0.2}}}
    patch = {
        "operations": [{"op": "remove", "path": "portfolio.constraints"}],
        "summary": "Remove constraints",
    }

    result = tool.apply_patch(config, patch)

    assert result.status == "error"
    assert "Risky changes detected" in (result.message or "")

    confirmed = tool.apply_patch(config, patch, confirm_risky=True)

    assert confirmed.status == "success"
    assert "constraints" not in confirmed.data["portfolio"]


def test_apply_patch_requires_confirmation_for_unknown_key_review() -> None:
    tool = ToolLayer()
    config = {"analysis": {"top_n": 10}}
    patch = {
        "operations": [{"op": "set", "path": "analysis.top_n", "value": 12}],
        "summary": "Update top_n",
        "needs_review": True,
    }

    result = tool.apply_patch(config, patch)

    assert result.status == "error"
    assert "UNKNOWN_KEYS" in (result.message or "")

    confirmed = tool.apply_patch(config, patch, confirm_risky=True)

    assert confirmed.status == "success"
    assert confirmed.data["analysis"]["top_n"] == 12
