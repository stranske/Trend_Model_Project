"""
Apply ranking triggers, enforce min/max constraints, and adjust weights.
(The real implementation will be generated later.)
"""

from typing import Mapping, Any
import pandas as pd


class Rebalancer:  # pylint: disable=too-few-public-methods
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg

    def apply_triggers(
        self, prev_weights: pd.Series, score_frame: pd.DataFrame
    ) -> pd.Series:
        """Stub — returns weights unchanged."""
        return prev_weights.copy()
