"""Core harness: turn a config + a scenario patch into normalized output.

The harness calls the *logic layer* (``api.run_simulation``) rather than the UI,
because baselines belong at the stable computational boundary, not the brittle
Streamlit surface. A "scenario" is just a base config plus a patch of dotted
keys (e.g. ``{"vol_adjust.target_vol": 0.20}``).

Everything here is app-specific glue for TMP. The generic ideas -- patch a
config, run, normalize output, capture runtime config-read coverage -- are what
will later be extracted into the shared kit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# Demo data is monthly (frequency: ME). Periods-per-year for annualization.
# When the kit grows beyond the monthly demo, infer this from the return index.
PERIODS_PER_YEAR = 12
_WEIGHT_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Config patching
# --------------------------------------------------------------------------- #
def _set_dotted(cfg: Any, dotted: str, value: Any) -> None:
    """Apply a single ``a.b.c = value`` patch to a loaded config.

    The first segment is a config *section* (an attribute on the Config object,
    usually a dict such as ``portfolio`` or ``vol_adjust``); remaining segments
    descend into nested dicts. A single-segment key (e.g. ``seed``) is set as a
    plain attribute.
    """
    parts = dotted.split(".")
    if len(parts) == 1:
        setattr(cfg, parts[0], value)
        return
    section = getattr(cfg, parts[0], None)
    if section is None or not isinstance(section, Mapping):
        # Materialize a dict section if missing so the patch is observable.
        section = {}
        setattr(cfg, parts[0], section)
    node: Any = section
    for key in parts[1:-1]:
        nxt = node.get(key)
        if not isinstance(nxt, Mapping):
            nxt = {}
            node[key] = nxt
        node = nxt
    node[parts[-1]] = value


def apply_patch(cfg: Any, patch: Mapping[str, Any] | None) -> Any:
    """Apply a dotted-key patch dict to a config, returning the same config."""
    for dotted, value in (patch or {}).items():
        _set_dotted(cfg, dotted, value)
    return cfg


def _resolve_csv_path(cfg: Any, config_path: Path) -> Path:
    """Resolve ``data.csv_path`` relative to the config file's directory."""
    raw = str(cfg.data.get("csv_path"))
    candidate = (config_path.parent / raw).resolve()
    if candidate.exists():
        return candidate
    # Fall back to repo-root-relative.
    return (REPO_ROOT / raw).resolve()


# --------------------------------------------------------------------------- #
# Normalized output
# --------------------------------------------------------------------------- #
@dataclass
class ScenarioOutput:
    """A normalized, comparison-friendly snapshot of one run's results."""

    metrics: pd.DataFrame
    weights: pd.Series
    fund_weights: pd.Series  # weights excluding benchmark columns
    turnover: pd.Series
    portfolio: pd.Series
    costs: dict[str, float]
    seed: int
    config_keys_read: set[str] = field(default_factory=set)
    selected_count: int | None = None
    declared_selected_count: int | None = None

    def derived(self, rf_annual: float = 0.0) -> dict[str, float]:
        """Economic summary stats computed from the constructed portfolio."""
        out: dict[str, float] = {}
        w = self.fund_weights
        out["weight_sum"] = float(w.sum())
        out["max_weight"] = float(w.max()) if len(w) else float("nan")
        out["min_weight"] = float(w.min()) if len(w) else float("nan")
        nonzero_count = int((w.abs() > _WEIGHT_EPS).sum())
        out["num_selected"] = (
            self.selected_count if self.selected_count is not None else nonzero_count
        )
        out["num_negative_weights"] = int((w < -_WEIGHT_EPS).sum())
        out["max_turnover"] = float(self.turnover.max()) if len(self.turnover) else float("nan")
        out.update(_ann_stats(self.portfolio, rf_annual=rf_annual))
        # Mean of the *reported* per-fund Sharpe (from the metrics table). Unlike
        # the portfolio-derived `sharpe` above, this reflects the rf override that
        # flows into the reported metrics, so it's the right lever for rf checks.
        if "sharpe" in self.metrics.columns:
            out["reported_sharpe"] = float(self.metrics["sharpe"].mean())
        return out


def _ann_stats(returns: pd.Series, rf_annual: float = 0.0) -> dict[str, float]:
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 2:
        return {
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }
    n = len(r)
    std = float(r.std(ddof=1))
    ann_return = float((1.0 + r).prod() ** (PERIODS_PER_YEAR / n) - 1.0)
    ann_vol = std * (PERIODS_PER_YEAR**0.5)
    rf_periodic = (1.0 + rf_annual) ** (1.0 / PERIODS_PER_YEAR) - 1.0
    sharpe = (
        ((float(r.mean()) - rf_periodic) / std) * (PERIODS_PER_YEAR**0.5)
        if std > 0
        else float("nan")
    )
    cum = (1.0 + r).cumprod()
    max_dd = float((cum / cum.cummax() - 1.0).min())
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
    }


def _as_series(value: Any) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.astype(float)
    if isinstance(value, Mapping):
        return pd.Series(value, dtype=float)
    return pd.Series(dtype=float)


def _benchmark_labels(cfg: Any) -> set[str]:
    bench = getattr(cfg, "benchmarks", {}) or {}
    if isinstance(bench, Mapping):
        return {str(v) for v in bench.values()}
    return set()


def _selected_fund_count(
    selected_funds: Any,
    fund_weights: pd.Series,
) -> int | None:
    if not isinstance(selected_funds, list):
        return None
    labels = [str(fund) for fund in selected_funds]
    matched = [label for label in labels if label in fund_weights.index]
    if not matched:
        return len(labels)
    return sum(1 for label in matched if abs(float(fund_weights[label])) > _WEIGHT_EPS)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run_scenario(
    config_path: str | Path = "config/demo.yml",
    patch: Mapping[str, Any] | None = None,
    *,
    track_reads: bool = True,
) -> ScenarioOutput:
    """Load a config, apply a patch, run the pipeline, and normalize the output.

    Runtime config-read coverage is captured best-effort via TMP's existing
    ``ConfigCoverageTracker``; if the config object refuses wrapping it simply
    comes back empty, which the manifest treats as "unknown" rather than failing.
    """
    from trend_analysis import api
    from trend_analysis.config import load
    from trend_analysis.data import load_csv

    cfg_path = (REPO_ROOT / config_path).resolve()
    cfg = load(str(cfg_path))
    apply_patch(cfg, patch)

    csv_path = _resolve_csv_path(cfg, cfg_path)
    df = load_csv(str(csv_path), errors="raise")
    if df is None:
        raise FileNotFoundError(csv_path)

    keys_read: set[str] = set()
    tracker = None
    if track_reads:
        try:
            from trend_analysis.config.coverage import (
                ConfigCoverageTracker,
                activate_config_coverage,
                deactivate_config_coverage,
                wrap_config_for_coverage,
            )

            tracker = ConfigCoverageTracker()
            activate_config_coverage(tracker)
            wrap_config_for_coverage(cfg, tracker)
        except Exception:
            tracker = None

    try:
        res = api.run_simulation(cfg, df)
    finally:
        if tracker is not None:
            try:
                deactivate_config_coverage()
                keys_read = set(tracker.generate_report().read)
            except Exception:
                keys_read = set()

    weights = _as_series(res.weights)
    bench = _benchmark_labels(cfg)
    fund_weights = weights[[i for i in weights.index if i not in bench]]
    costs = res.costs if isinstance(res.costs, dict) else {}

    selected_count = None
    declared_selected_count = None
    details = getattr(res, "details", None)
    if isinstance(details, Mapping):
        selected_funds = details.get("selected_funds")
        if isinstance(selected_funds, list):
            declared_selected_count = len(selected_funds)
        selected_count = _selected_fund_count(
            selected_funds,
            fund_weights,
        )
    period_results = getattr(res, "period_results", None)
    if not isinstance(period_results, list):
        if isinstance(details, Mapping):
            period_results = details.get("period_results")
    if selected_count is None and isinstance(period_results, list):
        for period_result in reversed(period_results):
            if not isinstance(period_result, Mapping):
                continue
            selected_count = _selected_fund_count(
                period_result.get("selected_funds"),
                fund_weights,
            )
            if selected_count is not None:
                break
            metadata = period_result.get("metadata")
            universe = metadata.get("universe") if isinstance(metadata, Mapping) else None
            if isinstance(universe, Mapping):
                selected_count_raw = universe.get("selected_count")
                if isinstance(selected_count_raw, int):
                    selected_count = selected_count_raw
                    break

    return ScenarioOutput(
        metrics=res.metrics.copy(),
        weights=weights,
        fund_weights=fund_weights,
        turnover=_as_series(res.turnover),
        portfolio=_as_series(res.portfolio),
        costs={str(k): float(v) for k, v in costs.items()},
        seed=int(getattr(res, "seed", 42)),
        config_keys_read=keys_read,
        selected_count=selected_count,
        declared_selected_count=declared_selected_count,
    )
