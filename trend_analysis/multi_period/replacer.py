"""
Very small “strike / replace” helper required by tests/test_replacer.py.

Rules implemented:

* Two consecutive z‑scores < ‑1.0  →  drop the fund
* Any un‑held fund with z‑score > +1.0  →  add the fund
* Surviving funds are re‑weighted equally
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


_LOW_Z = -1.0  # removal trigger
_LOW_STRIKES = 2  # consecutive strikes required
_HIGH_Z = 1.0  # addition trigger


class Rebalancer:  # pylint: disable=too-few-public-methods
    """Stub implementation that fulfils the Phase‑2 unit tests."""

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self._strikes: dict[str, int] = {}

    # ------------------------------------------------------------------
    def apply_triggers(
        self,
        prev_weights: Dict[str, float] | pd.Series,
        score_frame: pd.DataFrame,
    ) -> pd.Series:
        """
        Return next‑period weights after applying the two simple rules
        required by the unit tests.
        """
        prev_w = (
            prev_weights.copy()
            if isinstance(prev_weights, pd.Series)
            else pd.Series(prev_weights, dtype=float)
        )

        # --- 1) update strike counts & decide removals -----------------
        zscores = (
            score_frame["zscore"]
            if "zscore" in score_frame.columns
            else score_frame.iloc[:, 0]
        )
        to_drop: list[str] = []

        for f in prev_w.index:
            z = zscores.get(f, 0.0)
            if z < _LOW_Z:
                self._strikes[f] = self._strikes.get(f, 0) + 1
            else:
                self._strikes[f] = 0

            if self._strikes[f] >= _LOW_STRIKES:
                to_drop.append(f)

        for f in to_drop:
            prev_w.drop(labels=f, inplace=True, errors="ignore")
            self._strikes.pop(f, None)

        # --- 2) add hot sidelined funds --------------------------------
        for f, z in zscores.items():
            if f not in prev_w and z > _HIGH_Z:
                prev_w[f] = 0.0  # placeholder

        # --- 3) equal‑weight the survivors -----------------------------
        if prev_w.empty:
            return prev_w  # edge case

        eq = 1.0 / len(prev_w)
        prev_w[:] = eq
        return prev_w
