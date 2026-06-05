from __future__ import annotations

import trend.cli as unified_cli
import trend_analysis.cli as legacy_cli


def test_cli_helpers_are_shared_objects() -> None:
    assert unified_cli._apply_trend_spec_preset is legacy_cli._apply_trend_spec_preset
    assert unified_cli._apply_universe_mask is legacy_cli._apply_universe_mask
    assert unified_cli._attach_universe_paths is legacy_cli._attach_universe_paths
