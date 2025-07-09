"""
Apply ranking triggers, enforce min/max constraints, and adjust weights.
(The real implementation will be generated later.)
"""
from __future__ import annotations

from typing import Mapping
import pandas as pd


class Rebalancer:  # pylint: disable=too-few-public-methods
    def __init__(self, cfg: Mapping[str, object]) -> None:
        self.cfg = cfg

    def apply_triggers(
        self, prev_weights: Mapping[str, float], score_frame: pd.DataFrame
    ) -> dict[str, float]:
        """Stub — returns weights unchanged."""
        return dict(prev_weights)
