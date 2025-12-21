"""Very small “strike / replace” helper required by tests/test_replacer.py.

Rules implemented:

* Two consecutive z‑scores < ‑1.0  →  drop the fund
* Any un‑held fund with z‑score > +1.0  →  add the fund
* Surviving funds are re‑weighted equally
Phase‑2 placeholder: echoes the incoming weights so the rest of the pipeline
keeps running. Real strike / replacement logic comes later.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

_LOW_Z = -1.0  # removal trigger (soft)
_LOW_STRIKES = 2  # consecutive strikes required
_HIGH_Z = 1.0  # addition trigger (soft)


class Rebalancer:  # pylint: disable=too-few-public-methods
    """Stub implementation that fulfils the Phase‑2 unit tests."""

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self._strikes: dict[str, int] = {}
        self._entry_strikes: dict[str, int] = {}
        # Read thresholds from config if available (backward compatible).
        # Some configs historically placed threshold-hold knobs at the
        # portfolio root (e.g. portfolio.z_exit_soft) instead of under
        # portfolio.threshold_hold.*.
        portfolio = self.cfg.get("portfolio", {}) if isinstance(self.cfg, dict) else {}
        # Check if random selection mode - if so, disable z-score triggers
        selection_mode = portfolio.get("selection_mode", "rank")
        self.is_random_mode = selection_mode == "random"
        th = dict(portfolio.get("threshold_hold", {}) or {})
        for key in (
            "z_exit_soft",
            "z_exit_hard",
            "z_entry_soft",
            "z_entry_hard",
            "soft_strikes",
            "entry_soft_strikes",
            "entry_eligible_strikes",
            "target_n",
        ):
            if key not in th and key in portfolio:
                th[key] = portfolio[key]
        # Soft/hard exits and entries; default to legacy constants
        self.low_z_soft = float(th.get("z_exit_soft", _LOW_Z))
        self.low_z_hard = float(th["z_exit_hard"]) if "z_exit_hard" in th else None
        self.high_z_soft = float(th.get("z_entry_soft", _HIGH_Z))
        self.high_z_hard = float(th["z_entry_hard"]) if "z_entry_hard" in th else None
        self.soft_strikes = int(th.get("soft_strikes", _LOW_STRIKES))
        # Soft entry requires consecutive z >= z_entry_soft periods
        self.entry_soft_strikes = int(th.get("entry_soft_strikes", 1))
        # Eligible after this many consecutive soft-threshold periods; default one less than auto
        self.entry_eligible_strikes = int(
            th.get("entry_eligible_strikes", max(1, self.entry_soft_strikes - 1))
        )
        constraints = (
            portfolio.get("constraints", {}) if isinstance(portfolio, dict) else {}
        )
        mp_cfg = self.cfg.get("multi_period", {}) if isinstance(self.cfg, dict) else {}
        self.max_funds = int(constraints.get("max_funds", mp_cfg.get("max_funds", 10)))
        # Weighting behaviour for survivors during run_schedule
        th_name = (
            portfolio.get("threshold_hold", {}).get("weighting", "equal")
            if isinstance(portfolio, dict)
            else "equal"
        )
        self.weighting_name = str(th_name).lower()
        self.weighting_params = (
            portfolio.get("weighting", {}).get("params", {})
            if isinstance(portfolio, dict)
            else {}
        )

    # ------------------------------------------------------------------
    def apply_triggers(
        self,
        prev_weights: Dict[str, float] | pd.Series,
        score_frame: pd.DataFrame,
        *,
        random_seed: int | None = None,
        target_n: int | None = None,
    ) -> pd.Series:
        """Return next‑period weights after applying the two simple rules
        required by the unit tests.

        In random selection mode, a completely fresh random selection is made
        from the available universe each period. This is essential to avoid
        survivorship bias - we can only select from funds available at each
        point in time, not funds that we know will survive into the future.
        """
        import numpy as np

        prev_w = (
            prev_weights.copy()
            if isinstance(prev_weights, pd.Series)
            else pd.Series(prev_weights, dtype=float)
        )

        # In random mode, do a FRESH random selection each period from available universe
        if self.is_random_mode:
            # Get available funds from the score_frame (these are funds with data in this period)
            available = list(score_frame.index)
            if not available:
                return pd.Series(dtype=float)

            # Determine how many funds to select
            n_select = target_n if target_n is not None else self.max_funds
            n_select = max(
                1, min(n_select, len(available))
            )  # Ensure at least 1, at most len(available)

            # Fresh random selection each period
            # Ensure seed is a valid non-negative integer
            try:
                # Use provided seed if not None; otherwise default to 42
                raw_seed = random_seed if random_seed is not None else 42
                safe_seed = abs(int(raw_seed))
            except (TypeError, ValueError):
                # Fall back to default seed on any conversion error
                safe_seed = 42
            rng = np.random.default_rng(safe_seed)
            selected = list(rng.choice(available, size=n_select, replace=False))

            # Equal weight the randomly selected funds
            eq = 1.0 / len(selected) if selected else 0.0
            return pd.Series({f: eq for f in selected}, dtype=float)

        # --- 1) update strike counts & decide removals -----------------
        zscores = (
            score_frame["zscore"]
            if "zscore" in score_frame.columns
            else score_frame.iloc[:, 0]
        )
        to_drop: list[str] = []

        for f in prev_w.index:
            f_str = str(f)
            z = zscores.get(f_str, 0.0)
            # Hard exit: immediate drop if below hard threshold (only if configured)
            if self.low_z_hard is not None and z < self.low_z_hard:
                to_drop.append(f_str)
                self._strikes.pop(f_str, None)
                continue
            # Soft exit: accumulate strikes below soft threshold
            if z < self.low_z_soft:
                self._strikes[f_str] = self._strikes.get(f_str, 0) + 1
            else:
                self._strikes[f_str] = 0
            if self._strikes.get(f_str, 0) >= self.soft_strikes:
                to_drop.append(f_str)

        for f in to_drop:
            prev_w.drop(labels=f, inplace=True, errors="ignore")
            self._strikes.pop(f, None)

        # --- 2) add sidelined funds based on entry rules ----------------
        hard_cands: list[tuple[str, float]] = []
        auto_cands: list[tuple[str, float]] = []
        eligible_cands: list[tuple[str, float]] = []

        for f, z in zscores.items():
            f_str = str(f)
            if f_str in prev_w:
                continue
            # Update soft entry strike counts
            if z >= self.high_z_soft:
                self._entry_strikes[f_str] = self._entry_strikes.get(f_str, 0) + 1
            else:
                self._entry_strikes[f_str] = 0
            # Classify candidates
            if self.high_z_hard is not None and z >= self.high_z_hard:
                hard_cands.append((f_str, float(z)))
            elif self._entry_strikes.get(f_str, 0) >= self.entry_soft_strikes:
                auto_cands.append((f_str, float(z)))
            elif self._entry_strikes.get(f_str, 0) >= self.entry_eligible_strikes:
                eligible_cands.append((f_str, float(z)))

        # Add in priority order: hard > auto > eligible (all capacity-limited)
        for bucket in (
            hard_cands,
            sorted(auto_cands, key=lambda x: x[1], reverse=True),
            sorted(eligible_cands, key=lambda x: x[1], reverse=True),
        ):
            for f_str, _z in bucket:
                if f_str in prev_w:
                    continue
                if len(prev_w) >= self.max_funds:
                    break
                prev_w[f_str] = 0.0
                self._entry_strikes[f_str] = 0

        # --- 3) weight the survivors ----------------------------------
        if prev_w.empty:
            return prev_w  # edge case
        # If configured to use bayesian weighting, compute simple
        # score-proportional weights using the provided zscores (acts as a
        # proxy for the real bayes in tests).
        if self.weighting_name in {"score_prop_bayes", "bayes", "score_bayes"}:
            # zscores might be missing for some held funds (shouldn't in tests).
            # Fallback to 0.
            z = score_frame.get("zscore")
            if z is None:
                eq = 1.0 / len(prev_w)
                prev_w[:] = eq
                return prev_w
            # shift to positive and proportionally allocate
            z_held = z.reindex(prev_w.index).astype(float).fillna(0.0)
            # ensure all positive: add a constant so min becomes small positive
            shift = float(max(0.0, -z_held.min() + 1e-9))
            base = (z_held + shift).clip(lower=1e-9)
            w = base / float(base.sum())
            return w.astype(float)
        else:
            eq = 1.0 / len(prev_w)
            prev_w[:] = eq
            return prev_w
