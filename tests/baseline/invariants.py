"""Tier 3 economic invariants -- things that must hold for *every* scenario.

Each invariant returns an ``InvariantResult``. ``severity`` distinguishes:
  * "error" -- an economically impossible result; fails the suite.
  * "warn"  -- implausible but not impossible (e.g. Sharpe > 5 over a short
               window); reported, never fails. Per the confirmed plan, the
               Sharpe band is a soft flag applied to annualized figures only.

Bounds were proposed and confirmed for the TMP pilot; edit them here as the
economic understanding of the model sharpens.
"""

from __future__ import annotations

from baseline_kit import InvariantResult

from .harness import ScenarioOutput

# Confirmed bounds.
WEIGHT_SUM_TOL = 0.02          # weights should sum to ~1 (allow small slack)
MAX_WEIGHT_TOL = 1e-6          # tolerance when checking the max-weight cap
SHARPE_SOFT_CAP = 5.0          # soft flag: meaningful-length Sharpe rarely > 5
LEVERAGE_CAP_DEFAULT = 2.0     # gross exposure ceiling when not otherwise set


def check_all(
    out: ScenarioOutput,
    *,
    long_only: bool = True,
    max_weight: float | None = 0.25,
    min_funds: int | None = None,
    max_funds: int | None = None,
    leverage_cap: float = LEVERAGE_CAP_DEFAULT,
) -> list[InvariantResult]:
    d = out.derived()
    results: list[InvariantResult] = []

    def add(name, ok, detail, severity="error"):
        results.append(InvariantResult(name, bool(ok), severity, detail))

    # 1. Weights sum to ~1 (or gross exposure within leverage cap).
    gross = float(out.fund_weights.abs().sum())
    add(
        "weight_sum_near_one",
        abs(d["weight_sum"] - 1.0) <= WEIGHT_SUM_TOL or gross <= leverage_cap + WEIGHT_SUM_TOL,
        f"weight_sum={d['weight_sum']:.4f}, gross={gross:.4f}, cap={leverage_cap}",
    )

    # 2. long_only => no negative weights.
    if long_only:
        add(
            "no_negative_weights_when_long_only",
            d["num_negative_weights"] == 0,
            f"num_negative_weights={d['num_negative_weights']}",
        )

    # 3. Every weight <= max_weight cap (+tol).
    if max_weight is not None:
        add(
            "max_weight_respected",
            d["max_weight"] <= max_weight + MAX_WEIGHT_TOL,
            f"max_weight={d['max_weight']:.4f}, cap={max_weight}",
        )

    # 4. Selected fund count within [min_funds, max_funds].
    if min_funds is not None:
        add(
            "at_least_min_funds",
            d["num_selected"] >= min_funds,
            f"num_selected={d['num_selected']}, min={min_funds}",
        )
    if max_funds is not None:
        add(
            "at_most_max_funds",
            d["num_selected"] <= max_funds,
            f"num_selected={d['num_selected']}, max={max_funds}",
        )

    # 5. Volatility >= 0; max drawdown in [-1, 0].
    add("vol_non_negative", d["ann_vol"] >= 0, f"ann_vol={d['ann_vol']:.4f}")
    add(
        "drawdown_in_range",
        -1.0 - 1e-9 <= d["max_drawdown"] <= 1e-9,
        f"max_drawdown={d['max_drawdown']:.4f}",
    )

    # 6. Sharpe finite and within a plausible band (soft warning only).
    import math

    sharpe = d["sharpe"]
    add(
        "sharpe_finite",
        math.isfinite(sharpe),
        f"sharpe={sharpe}",
    )
    add(
        "sharpe_within_soft_band",
        abs(sharpe) <= SHARPE_SOFT_CAP if math.isfinite(sharpe) else True,
        f"sharpe={sharpe:.4f}, soft_cap=±{SHARPE_SOFT_CAP}",
        severity="warn",
    )

    # 7. No NaN/inf in reported metrics frame.
    bad = (~out.metrics.apply(lambda c: c.map(_is_finite_or_na))).to_numpy().sum()
    add(
        "no_inf_in_metrics",
        bad == 0,
        f"non-finite metric cells={int(bad)}",
    )

    return results


def _is_finite_or_na(x) -> bool:
    import math

    import pandas as pd

    if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
        return True  # NaN is allowed (missing), inf is not
    if isinstance(x, (int, float)):
        return math.isfinite(x)
    return True  # non-numeric cells (labels) are fine
