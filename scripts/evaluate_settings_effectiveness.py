#!/usr/bin/env python3
"""Evaluate whether Streamlit model settings affect simulation outputs."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PAGE = PROJECT_ROOT / "streamlit_app" / "pages" / "2_Model.py"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_settings_wiring as wiring  # noqa: E402


def _shift_numeric(value: Any, delta: float) -> float:
    try:
        return float(value) + delta
    except (TypeError, ValueError):
        return delta


def _cycle_choice(value: Any, choices: Iterable[str]) -> str:
    choices_list = list(choices)
    value_str = str(value)
    if value_str in choices_list:
        idx = (choices_list.index(value_str) + 1) % len(choices_list)
        return choices_list[idx]
    return choices_list[0]


def _pick_date(returns: pd.DataFrame, ratio: float) -> str:
    index = returns.index
    if index.empty:
        return ""
    pos = int(max(min(len(index) - 1, int(len(index) * ratio)), 0))
    return pd.Timestamp(index[pos]).strftime("%Y-%m-%d")


MODE_CONTEXT_OVERRIDES: dict[str, dict[str, Any]] = {
    "buy_hold_initial": {"inclusion_approach": "buy_and_hold"},
    "rank_pct": {"inclusion_approach": "top_pct"},
    "rf_rate_annual": {"rf_override_enabled": True},
    "info_ratio_benchmark": {"metric_weights": {"info_ratio": 1.0}},
    "min_weight": {"weighting_scheme": "risk_parity"},
    "shrinkage_enabled": {"weighting_scheme": "robust_mv"},
    "shrinkage_method": {"weighting_scheme": "robust_mv"},
    "sticky_add_periods": {"multi_period_enabled": True},
    "sticky_drop_periods": {"multi_period_enabled": True},
    "z_entry_soft": {"multi_period_enabled": True},
    "z_exit_soft": {"multi_period_enabled": True},
    "soft_strikes": {"multi_period_enabled": True},
    "entry_soft_strikes": {"multi_period_enabled": True},
    "min_weight_strikes": {"multi_period_enabled": True},
    "z_entry_hard": {"multi_period_enabled": True},
    "z_exit_hard": {"multi_period_enabled": True},
    "mp_min_funds": {"multi_period_enabled": True},
    "mp_max_funds": {"multi_period_enabled": True},
}

DEFAULT_OVERRIDES: dict[str, Any] = {
    "preset": "Baseline",
    "info_ratio_benchmark": "",
    "rf_override_enabled": False,
    "vol_adjust_enabled": True,
    "vol_window_length": 63,
    "vol_window_decay": "ewma",
    "vol_ewma_lambda": 0.94,
    "buy_hold_initial": "top_n",
    "report_regime_analysis": False,
    "report_concentration": True,
    "report_benchmark_comparison": True,
    "report_factor_exposures": False,
    "report_attribution": False,
    "report_rolling_metrics": True,
}

VARIATION_OVERRIDES: dict[str, Callable[[Any, pd.DataFrame], Any]] = {
    "inclusion_approach": lambda _v, _r: "top_n",
    "buy_hold_initial": lambda _v, _r: "threshold",
    "date_mode": lambda _v, _r: "explicit",
    "start_date": lambda _v, r: _pick_date(r, 0.1),
    "end_date": lambda _v, r: _pick_date(r, 0.9),
    "rank_pct": lambda _v, _r: 0.2,
    "weighting_scheme": lambda _v, _r: "risk_parity",
    "risk_target": lambda v, _r: _shift_numeric(v, 0.05),
    "vol_floor": lambda v, _r: _shift_numeric(v, 0.01),
    "vol_window_length": lambda _v, _r: 21,
    "vol_window_decay": lambda _v, _r: "simple",
    "vol_ewma_lambda": lambda v, _r: 0.85 if v and float(v) > 0.9 else 0.94,
    "max_weight": lambda v, _r: max(float(v) - 0.05, 0.05),
    "min_weight": lambda v, _r: min(float(v) + 0.02, 0.25),
    "rebalance_freq": lambda v, _r: _cycle_choice(v, ["M", "Q", "A"]),
    "transaction_cost_bps": lambda _v, _r: 10,
    "trend_window": lambda _v, _r: 126,
    "trend_lag": lambda v, _r: max(int(v or 1) + 1, 1),
    "trend_zscore": lambda v, _r: not bool(v),
    "trend_vol_adjust": lambda v, _r: not bool(v),
    "trend_vol_target": lambda v, _r: 0.1 if v in (None, "") else None,
    "regime_enabled": lambda v, _r: not bool(v),
    "regime_proxy": lambda v, _r: "AGG" if str(v).upper() != "AGG" else "SPX",
    "shrinkage_enabled": lambda v, _r: not bool(v),
    "shrinkage_method": lambda v, _r: _cycle_choice(v, ["ledoit_wolf", "oas", "none"]),
    "random_seed": lambda v, _r: int(v or 42) + 7,
    "condition_threshold": lambda v, _r: float(v) * 0.1,
    "safe_mode": lambda v, _r: "risk_parity" if v != "risk_parity" else "hrp",
    "long_only": lambda v, _r: not bool(v),
    "z_entry_soft": lambda v, _r: float(v) + 0.5,
    "z_exit_soft": lambda v, _r: float(v) - 0.5,
    "soft_strikes": lambda v, _r: int(v) + 1,
    "entry_soft_strikes": lambda v, _r: int(v) + 1,
    "min_weight_strikes": lambda v, _r: int(v) + 1,
    "sticky_add_periods": lambda v, _r: int(v) + 1,
    "sticky_drop_periods": lambda v, _r: int(v) + 1,
    "ci_level": lambda v, _r: 0.95 if float(v or 0.0) < 0.5 else 0.0,
    "multi_period_enabled": lambda v, _r: not bool(v),
    "multi_period_frequency": lambda v, _r: _cycle_choice(v, ["M", "Q", "A"]),
    "slippage_bps": lambda _v, _r: 5,
    "bottom_k": lambda v, _r: int(v) + 1,
    "mp_min_funds": lambda v, _r: max(int(v) - 2, 1),
    "mp_max_funds": lambda v, _r: int(v) + 2,
    "z_entry_hard": lambda v, _r: 2.0 if v in (None, "") else None,
    "z_exit_hard": lambda v, _r: -2.0 if v in (None, "") else None,
    "report_regime_analysis": lambda v, _r: not bool(v),
    "report_concentration": lambda v, _r: not bool(v),
    "report_benchmark_comparison": lambda v, _r: not bool(v),
    "report_factor_exposures": lambda v, _r: not bool(v),
    "report_attribution": lambda v, _r: not bool(v),
    "report_rolling_metrics": lambda v, _r: not bool(v),
}


@dataclass
class SettingEvaluation:
    name: str
    baseline_value: Any
    test_value: Any
    status: str
    diff_metrics: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _extract_dict_keys(node: ast.AST) -> set[str]:
    if not isinstance(node, ast.Dict):
        return set()
    keys: set[str] = set()
    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.add(key.value)
    return keys


def _extract_subscript_key(node: ast.Subscript) -> str | None:
    key_node = node.slice
    if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
        return key_node.value
    return None


def _is_model_state_ref(node: ast.AST) -> bool:
    if isinstance(node, ast.Name) and node.id == "model_state":
        return True
    if isinstance(node, ast.Subscript):
        key = _extract_subscript_key(node)
        if key != "model_state":
            return False
        if isinstance(node.value, ast.Attribute):
            return (
                isinstance(node.value.value, ast.Name)
                and node.value.value.id == "st"
                and node.value.attr == "session_state"
            )
    return False


def _extract_model_state_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Subscript):
        if _is_model_state_ref(node.value):
            return _extract_subscript_key(node)
    return None


def extract_settings_from_model_page(
    path: Path,
) -> tuple[dict[str, Any], set[str]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    preset_configs: dict[str, Any] = {}
    candidate_keys: set[str] = set()
    initial_state_keys: set[str] = set()
    model_state_keys: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PRESET_CONFIGS":
                    try:
                        preset_configs = ast.literal_eval(node.value)
                    except Exception:
                        preset_configs = {}
                if isinstance(target, ast.Name) and target.id == "candidate_state":
                    candidate_keys |= _extract_dict_keys(node.value)
                key = _extract_model_state_key(target)
                if key:
                    model_state_keys.add(key)
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and _is_model_state_ref(node.func.value)
            ):
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    model_state_keys.add(arg.value)
        if isinstance(node, ast.FunctionDef) and node.name == "_initial_model_state":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Return):
                    initial_state_keys |= _extract_dict_keys(inner.value)

    baseline_preset = preset_configs.get("Baseline", {})
    all_keys = set(baseline_preset.keys())
    all_keys |= candidate_keys
    all_keys |= initial_state_keys
    all_keys |= model_state_keys
    return baseline_preset, all_keys


def build_baseline_state(
    baseline_preset: dict[str, Any], settings: Iterable[str]
) -> dict[str, Any]:
    state = copy.deepcopy(baseline_preset)
    state.setdefault("metric_weights", {}).update(
        baseline_preset.get("metric_weights", {})
    )
    state.setdefault("preset", "Baseline")
    for key, value in DEFAULT_OVERRIDES.items():
        state.setdefault(key, value)
    for setting in settings:
        state.setdefault(setting, DEFAULT_OVERRIDES.get(setting))
    return state


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 12
    diffs = np.diff(index.values).astype("timedelta64[D]").astype(int)
    median_days = float(np.median(diffs)) if len(diffs) else 30.0
    if median_days <= 2:
        return 252
    if median_days <= 8:
        return 52
    return 12


def _average_period_weights(period_results: list[dict[str, Any]]) -> pd.Series | None:
    weight_maps = []
    for period in period_results:
        weights = period.get("weights")
        if isinstance(weights, dict) and weights:
            weight_maps.append(weights)
    if not weight_maps:
        return None
    funds = sorted({key for weights in weight_maps for key in weights})
    data = {fund: [] for fund in funds}
    for weights in weight_maps:
        for fund in funds:
            data[fund].append(float(weights.get(fund, 0.0)))
    return pd.Series({fund: float(np.mean(vals)) for fund, vals in data.items()})


def _compute_series_stats(series: pd.Series, periods_per_year: int) -> dict[str, Any]:
    returns = series.dropna()
    if returns.empty:
        return {}
    mean = float(returns.mean())
    vol = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (mean / vol * math.sqrt(periods_per_year)) if vol else 0.0
    cumulative = float((1 + returns).prod() - 1)
    return {
        "mean_return": mean,
        "volatility": vol,
        "sharpe": sharpe,
        "cumulative_return": cumulative,
    }


def _compute_turnover(result: Any) -> float | None:
    if getattr(result, "turnover", None) is not None:
        series = result.turnover
        if isinstance(series, pd.Series) and not series.empty:
            return float(series.mean())
    period_results = getattr(result, "period_results", None)
    if isinstance(period_results, list):
        turnovers = [
            float(p.get("turnover", 0.0)) for p in period_results if isinstance(p, dict)
        ]
        if turnovers:
            return float(np.mean(turnovers))
    return None


def _compute_weight_stats(result: Any) -> pd.Series | None:
    weights = getattr(result, "weights", None)
    if isinstance(weights, pd.Series) and not weights.empty:
        return weights.astype(float)
    period_results = getattr(result, "period_results", None)
    if isinstance(period_results, list):
        return _average_period_weights(period_results)
    return None


def _choose_explicit_dates(returns: pd.DataFrame) -> tuple[str, str]:
    index = returns.index
    if len(index) < 3:
        start = index.min()
        end = index.max()
    else:
        start = index[int(len(index) * 0.2)]
        end = index[int(len(index) * 0.8)]
    return pd.Timestamp(start).strftime("%Y-%m-%d"), pd.Timestamp(end).strftime(
        "%Y-%m-%d"
    )


def _generate_variant(setting: str, baseline: Any, returns: pd.DataFrame) -> Any:
    override = VARIATION_OVERRIDES.get(setting)
    if override:
        return override(baseline, returns)
    if isinstance(baseline, bool):
        return not baseline
    if isinstance(baseline, int):
        return baseline + 1 if baseline >= 0 else 1
    if isinstance(baseline, float):
        return baseline * 1.2 if baseline else 0.1
    if isinstance(baseline, str):
        return f"{baseline}_alt"
    if baseline is None and setting in {"start_date", "end_date"}:
        start, end = _choose_explicit_dates(returns)
        return start if setting == "start_date" else end
    return baseline


def _apply_context_overrides(
    setting: str, base_state: dict[str, Any], test_state: dict[str, Any]
) -> None:
    overrides = MODE_CONTEXT_OVERRIDES.get(setting, {})
    for key, value in overrides.items():
        if key == "metric_weights" and isinstance(value, dict):
            base_state.setdefault("metric_weights", {}).update(value)
            test_state.setdefault("metric_weights", {}).update(value)
        else:
            base_state[key] = value
            test_state[key] = value


def _apply_date_mode_overrides(
    setting: str,
    base_state: dict[str, Any],
    test_state: dict[str, Any],
    returns: pd.DataFrame,
) -> None:
    if setting in {"start_date", "end_date"}:
        base_state["date_mode"] = "explicit"
        test_state["date_mode"] = "explicit"
        start, end = _choose_explicit_dates(returns)
        base_state["start_date"] = start
        base_state["end_date"] = end
        test_state["start_date"] = start
        test_state["end_date"] = end
    if setting == "date_mode":
        start, end = _choose_explicit_dates(returns)
        test_state["start_date"] = start
        test_state["end_date"] = end


def evaluate_setting(
    name: str, baseline_state: dict[str, Any], returns: pd.DataFrame
) -> SettingEvaluation:
    base_state = copy.deepcopy(baseline_state)
    test_state = copy.deepcopy(baseline_state)

    _apply_context_overrides(name, base_state, test_state)
    _apply_date_mode_overrides(name, base_state, test_state, returns)

    baseline_value = base_state.get(name)
    if baseline_value is None and name in DEFAULT_OVERRIDES:
        baseline_value = DEFAULT_OVERRIDES[name]
        base_state[name] = baseline_value

    test_value = _generate_variant(name, baseline_value, returns)
    test_state[name] = test_value

    if name == "metric_weights":
        base_weights = base_state.get("metric_weights", {})
        test_weights = copy.deepcopy(base_weights)
        test_weights["sharpe"] = float(test_weights.get("sharpe", 1.0)) + 0.5
        test_state["metric_weights"] = test_weights
        test_value = test_weights

    try:
        baseline_result = wiring.run_analysis_with_state(returns, base_state)
        test_result = wiring.run_analysis_with_state(returns, test_state)

        periods_per_year = infer_periods_per_year(returns.index)
        baseline_returns = getattr(baseline_result, "portfolio", None)
        test_returns = getattr(test_result, "portfolio", None)

        baseline_stats = (
            _compute_series_stats(baseline_returns, periods_per_year)
            if isinstance(baseline_returns, pd.Series)
            else {}
        )
        test_stats = (
            _compute_series_stats(test_returns, periods_per_year)
            if isinstance(test_returns, pd.Series)
            else {}
        )

        baseline_turnover = _compute_turnover(baseline_result)
        test_turnover = _compute_turnover(test_result)

        base_weights = _compute_weight_stats(baseline_result)
        test_weights = _compute_weight_stats(test_result)

        weight_l1 = None
        weight_l2 = None
        weight_max = None
        if base_weights is not None and test_weights is not None:
            aligned = pd.concat([base_weights, test_weights], axis=1).fillna(0.0)
            diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
            weight_l1 = float(diff.abs().sum())
            weight_l2 = float(np.sqrt((diff**2).sum()))
            weight_max = float(diff.abs().max())

        t_stat = None
        p_value = None
        if isinstance(baseline_returns, pd.Series) and isinstance(
            test_returns, pd.Series
        ):
            aligned_returns = pd.concat(
                [baseline_returns, test_returns], axis=1
            ).dropna()
            if len(aligned_returns) >= 3:
                t_stat, p_value = stats.ttest_rel(
                    aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1]
                )
                if p_value is not None and np.isnan(p_value):
                    p_value = None

        diff_metrics = {
            "weight_l1": weight_l1,
            "weight_l2": weight_l2,
            "weight_max_abs": weight_max,
            "mean_return_diff": _diff(
                baseline_stats.get("mean_return"), test_stats.get("mean_return")
            ),
            "volatility_diff": _diff(
                baseline_stats.get("volatility"), test_stats.get("volatility")
            ),
            "sharpe_diff": _diff(
                baseline_stats.get("sharpe"), test_stats.get("sharpe")
            ),
            "cumulative_return_diff": _diff(
                baseline_stats.get("cumulative_return"),
                test_stats.get("cumulative_return"),
            ),
            "turnover_diff": _diff(baseline_turnover, test_turnover),
        }

        effect_thresholds = {
            "weight_l1": 0.02,
            "mean_return_diff": 1e-3,
            "sharpe_diff": 0.05,
            "turnover_diff": 1e-3,
            "cumulative_return_diff": 1e-3,
        }
        effect_detected = any(
            diff_metrics.get(key) is not None
            and abs(float(diff_metrics[key])) >= effect_thresholds[key]
            for key in effect_thresholds
        )
        if p_value is not None and p_value < 0.05:
            effect_detected = True

        status = "EFFECTIVE" if effect_detected else "NO_EFFECT"

        return SettingEvaluation(
            name=name,
            baseline_value=baseline_value,
            test_value=test_value,
            status=status,
            diff_metrics=diff_metrics,
            stats={
                "baseline": baseline_stats,
                "test": test_stats,
                "t_stat": None if t_stat is None else float(t_stat),
                "p_value": None if p_value is None else float(p_value),
            },
            details={
                "baseline_turnover": baseline_turnover,
                "test_turnover": test_turnover,
            },
        )

    except Exception as exc:
        return SettingEvaluation(
            name=name,
            baseline_value=baseline_value,
            test_value=test_value,
            status="ERROR",
            error=str(exc),
            details={"traceback": traceback.format_exc()},
        )


def _diff(a: Any, b: Any) -> float | None:
    try:
        if a is None or b is None:
            return None
        return float(b) - float(a)
    except (TypeError, ValueError):
        return None


def run_evaluation(
    returns: pd.DataFrame, settings: list[str], baseline_state: dict[str, Any]
) -> list[SettingEvaluation]:
    results: list[SettingEvaluation] = []
    for idx, setting in enumerate(settings, 1):
        print(f"[{idx}/{len(settings)}] Evaluating {setting}...")
        results.append(evaluate_setting(setting, baseline_state, returns))
    return results


def save_results(
    results: list[SettingEvaluation], output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"settings_effectiveness_{timestamp}.json"
    csv_path = output_dir / f"settings_effectiveness_{timestamp}.csv"

    payload = [result.__dict__ for result in results]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    for result in results:
        row = {
            "setting": result.name,
            "baseline_value": json.dumps(result.baseline_value),
            "test_value": json.dumps(result.test_value),
            "status": result.status,
            "error": result.error or "",
        }
        row.update({f"diff_{k}": v for k, v in result.diff_metrics.items()})
        row.update(
            {
                "p_value": result.stats.get("p_value"),
                "t_stat": result.stats.get("t_stat"),
            }
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Directory for output JSON/CSV results",
    )
    args = parser.parse_args()

    print("Loading demo data...")
    returns = wiring.load_demo_data()
    print(f"Loaded {len(returns)} rows, {len(returns.columns)} columns")

    baseline_preset, settings_from_ui = extract_settings_from_model_page(MODEL_PAGE)
    settings = sorted(settings_from_ui | set(baseline_preset.keys()))
    baseline_state = build_baseline_state(baseline_preset, settings)

    results = run_evaluation(returns, settings, baseline_state)
    json_path, csv_path = save_results(results, args.output_dir)

    effective = sum(1 for r in results if r.status == "EFFECTIVE")
    errors = sum(1 for r in results if r.status == "ERROR")
    print(f"Completed {len(results)} settings: {effective} effective, {errors} errors.")
    print(f"Saved JSON report to {json_path}")
    print(f"Saved CSV report to {csv_path}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
