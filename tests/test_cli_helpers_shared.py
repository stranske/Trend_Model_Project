from __future__ import annotations

import trend.cli as unified_cli
import trend.cli_helpers as cli_helpers


def test_cli_uses_shared_helper_objects() -> None:
    assert unified_cli._apply_trend_spec_preset is cli_helpers._apply_trend_spec_preset
    assert unified_cli._apply_universe_mask is cli_helpers._apply_universe_mask
    assert unified_cli._attach_universe_paths is cli_helpers._attach_universe_paths
