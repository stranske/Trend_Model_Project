#!/usr/bin/env python3
"""Evaluate Streamlit settings effectiveness by comparing paired simulations."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.results import Results  # noqa: E402
from trend_analysis.api import RunResult, run_simulation  # noqa: E402

MODEL_FILE = PROJECT_ROOT / "streamlit_app" / "pages" / "2_Model.py"
MODEL_PAGE = MODEL_FILE  # Backward-compatible alias
TEST_WIRING_FILE = PROJECT_ROOT / "scripts" / "test_settings_wiring.py"

REPORTING_ONLY_PREFIXES = ("report_",)
REPORTING_ONLY_KEYS = {"ci_level"}

DEFAULT_REPORTING_STATE = {
    "report_regime_analysis": False,
    "report_concentration": True,
    "report_benchmark_comparison": True,
    "report_factor_exposures": False,
    "report_attribution": False,
    "report_rolling_metrics": True,
}

BASELINE_OVERRIDES = {
    "preset": "Baseline",
    "info_ratio_benchmark": "",
    "rf_override_enabled": False,
    "min_weight_strikes": 2,
    "buy_hold_initial": "top_n",
    "multi_period_enabled": True,
}

OPTIONS_BY_KEY = {
    "weighting_scheme": [
        "equal",
        "risk_parity",
        "hrp",
        "erc",
        "robust_mv",
        "robust_risk_parity",
    ],
    "inclusion_approach": ["threshold", "top_n", "top_pct", "random", "buy_and_hold"],
    "buy_hold_initial": ["top_n", "threshold", "top_pct", "random"],
    "multi_period_frequency": ["A", "Q", "M"],
    "rebalance_freq": ["M", "Q", "A"],
    "vol_window_decay": ["ewma", "simple"],
    "safe_mode": ["hrp", "risk_parity", "equal"],
    "shrinkage_method": ["ledoit_wolf", "oas", "none"],
    "date_mode": ["relative", "explicit"],
}

VARIATION_OVERRIDES: dict[str, Any] = {
    "lookback_periods": 6,
    "min_history_periods": 6,
    "evaluation_periods": 2,
    "selection_count": 5,
    "rank_pct": 0.30,
    "risk_target": 0.15,
    "vol_floor": 0.03,
    "vol_window_length": 21,
    "vol_ewma_lambda": 0.8,
    "max_weight": 0.10,
    "min_weight": 0.08,
    "cooldown_periods": 2,
    "max_turnover": 0.3,
    "transaction_cost_bps": 25,
    "slippage_bps": 10,
    "warmup_periods": 6,
    "random_seed": 123,
    "condition_threshold": 1.0e10,
    "preset": "Conservative",
    "regime_proxy": "RF",
    "min_tenure_periods": 6,
    "max_changes_per_period": 2,
    "max_active_positions": 8,
    "trend_window": 126,
    "trend_lag": 2,
    "z_entry_soft": 1.5,
    "z_exit_soft": -0.5,
    "soft_strikes": 3,
    "entry_soft_strikes": 2,
    "z_entry_hard": 1.5,
    "z_exit_hard": -1.5,
    "bottom_k": 2,
    "mp_min_funds": 8,
    "mp_max_funds": 18,
}

MODE_CONTEXT: dict[str, dict[str, Any]] = {
    "buy_hold_initial": {"inclusion_approach": "buy_and_hold"},
    "rank_pct": {"inclusion_approach": "top_pct"},
    "shrinkage_enabled": {"weighting_scheme": "robust_mv"},
    "shrinkage_method": {"weighting_scheme": "robust_mv"},
    "sticky_add_periods": {"multi_period_enabled": True},
    "sticky_drop_periods": {"multi_period_enabled": True},
    "rf_rate_annual": {"rf_override_enabled": True},
    "info_ratio_benchmark": {"metric_weights": {"info_ratio": 1.0}},
    "regime_proxy": {"regime_enabled": True},
}


@dataclass
class SettingResult:
    setting: str
    baseline_value: Any
    test_value: Any
    status: str
    mode_specific: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    required_context: dict[str, Any] = field(default_factory=dict)


def _extract_literal_dict(node: ast.AST) -> dict[str, Any] | None:
    try:
        value = ast.literal_eval(node)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _extract_literal_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Str):
        return node.s
    return None


def _load_setting_categories(file_path: Path) -> dict[str, str]:
    if not file_path.exists():
        return {}

    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return {}

    category_map: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "SETTINGS_TO_TEST" not in targets:
                continue
            if not isinstance(node.value, (ast.List, ast.Tuple)):
                break
            for elt in node.value.elts:
                if not isinstance(elt, ast.Call):
                    continue
                name_val = None
                category_val = None
                for kw in elt.keywords:
                    if kw.arg == "name":
                        name_val = _extract_literal_str(kw.value)
                    elif kw.arg == "category":
                        category_val = _extract_literal_str(kw.value)
                if name_val is None and elt.args:
                    name_val = _extract_literal_str(elt.args[0])
                if category_val is None and len(elt.args) > 3:
                    category_val = _extract_literal_str(elt.args[3])
                if name_val and category_val and name_val not in category_map:
                    category_map[name_val] = category_val
            break
    return category_map


def _recommendation_for_result(result: SettingResult) -> str:
    if result.setting.startswith(REPORTING_ONLY_PREFIXES) or (
        result.setting in REPORTING_ONLY_KEYS
    ):
        return "Reporting-only setting; no simulation impact expected."
    if result.required_context:
        context = ", ".join(
            f"{key}={value}" for key, value in sorted(result.required_context.items())
        )
        return f"Mode-specific setting. Ensure required context is set ({context})."
    setting = result.setting.lower()
    if "mode" in setting or "approach" in setting:
        return "Verify prerequisite settings align with the selected mode."
    if "weight" in setting:
        return "Check weighting logic in metrics.py or portfolio construction."
    if "window" in setting or "period" in setting:
        return "Ensure this setting flows into rolling window calculations."
    return "Confirm the setting is wired from UI state into pipeline inputs."


def _keys_from_dict(node: ast.Dict) -> set[str]:
    keys: set[str] = set()
    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.add(key.value)
    return keys


def _extract_settings_from_model(file_path: Path) -> set[str]:
    text = file_path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    settings: set[str] = set()

    # Extract baseline preset keys for defaults.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "PRESET_CONFIGS" in targets:
                preset_dict = _extract_literal_dict(node.value)
                if preset_dict and isinstance(preset_dict.get("Baseline"), dict):
                    settings.update(preset_dict["Baseline"].keys())
                elif isinstance(node.value, ast.Dict):
                    baseline_dict = None
                    for key_node, value_node in zip(node.value.keys, node.value.values):
                        if (
                            isinstance(key_node, ast.Constant)
                            and key_node.value == "Baseline"
                            and isinstance(value_node, ast.Dict)
                        ):
                            baseline_dict = value_node
                            break
                    if baseline_dict is not None:
                        settings.update(_keys_from_dict(baseline_dict))
                break

    # Extract keys from _initial_model_state return dict.
    for walk_node in ast.walk(tree):
        if (
            isinstance(walk_node, ast.FunctionDef)
            and walk_node.name == "_initial_model_state"
        ):
            for child in ast.walk(walk_node):
                if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
                    settings.update(_keys_from_dict(child.value))
                    break

    # Extract keys from candidate_state assignment.
    for assign_node in ast.walk(tree):
        if isinstance(assign_node, ast.Assign):
            targets = [t.id for t in assign_node.targets if isinstance(t, ast.Name)]
            if "candidate_state" in targets and isinstance(assign_node.value, ast.Dict):
                settings.update(_keys_from_dict(assign_node.value))

    # Regex scan for model_state accessors.
    for pattern in (
        r"model_state\.get\(\s*[\"']([^\"']+)[\"']",
        r"model_state\[\s*[\"']([^\"']+)[\"']\s*\]",
        r"session_state\[\s*[\"']model_state[\"']\s*\]\s*\[\s*[\"']([^\"']+)[\"']\s*\]",
    ):
        settings.update(re.findall(pattern, text))

    return settings


def _extract_baseline_preset(file_path: Path) -> dict[str, Any]:
    text = file_path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "PRESET_CONFIGS" in targets:
                preset_dict = _extract_literal_dict(node.value)
                if preset_dict and isinstance(preset_dict.get("Baseline"), dict):
                    return dict(preset_dict["Baseline"])
    return {}


def extract_settings_from_model_page(
    file_path: Path | str,
) -> tuple[dict[str, Any], set[str]]:
    """Return baseline settings and all discovered keys from a model page."""

    path = Path(file_path)
    baseline = _extract_baseline_preset(path)
    keys = _extract_settings_from_model(path)
    return baseline, keys


def _build_baseline_state(settings: Iterable[str]) -> dict[str, Any]:
    baseline = _extract_baseline_preset(MODEL_FILE)
    state = dict(baseline)
    state.update(BASELINE_OVERRIDES)
    state.update(DEFAULT_REPORTING_STATE)

    for key in settings:
        state.setdefault(key, None)
    return state


def _next_option(key: str, current: str | None) -> str | None:
    options = OPTIONS_BY_KEY.get(key)
    if not options:
        return None
    if current in options:
        idx = options.index(current)
        return options[(idx + 1) % len(options)]
    return options[0]


def _default_variation(key: str, value: Any) -> Any:
    if key in VARIATION_OVERRIDES:
        return VARIATION_OVERRIDES[key]
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        if value == 0:
            return 1
        delta = max(1, int(round(value * 0.3)))
        return max(1, value - delta)
    if isinstance(value, float):
        if value == 0.0:
            return 0.05
        return value * 1.5 if value < 1 else value + 0.5
    if isinstance(value, str):
        return _next_option(key, value)
    if isinstance(value, dict):
        if key == "metric_weights":
            return {"sharpe": 0.0, "return_ann": 2.0, "drawdown": 0.0}
    return None


def _resolve_variation(
    key: str, base_value: Any, returns: pd.DataFrame
) -> tuple[Any, dict[str, Any]]:
    required_context = dict(MODE_CONTEXT.get(key, {}))
    test_value = _default_variation(key, base_value)

    if key == "metric_weights":
        test_value = {"sharpe": 0.0, "return_ann": 2.0, "drawdown": 0.0}
    if key == "info_ratio_benchmark":
        test_value = "SPX"
    if key == "date_mode":
        test_value = "explicit"
    if key == "start_date":
        test_value = (returns.index.min() + pd.DateOffset(months=6)).strftime(
            "%Y-%m-%d"
        )
        required_context["date_mode"] = "explicit"
    if key == "end_date":
        test_value = (returns.index.max() - pd.DateOffset(months=6)).strftime(
            "%Y-%m-%d"
        )
        required_context["date_mode"] = "explicit"
    if key == "trend_min_periods":
        test_value = 10 if base_value in (None, "", 0) else None
    if key == "trend_vol_target":
        test_value = 0.12 if base_value in (None, "", 0) else None

    return test_value, required_context


def _apply_context(state: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    merged = dict(state)
    for key, value in context.items():
        if key == "metric_weights":
            merged.setdefault("metric_weights", {})
            merged["metric_weights"] = {**merged["metric_weights"], **value}
        else:
            merged[key] = value
    return merged


def _prepare_returns(df: pd.DataFrame) -> pd.DataFrame:
    reset = df.reset_index()
    index_name = df.index.name or "Date"
    return reset.rename(columns={index_name: "Date"})


def _run_simulation(
    returns: pd.DataFrame, model_state: dict[str, Any], benchmark: str | None
) -> RunResult:
    from streamlit_app.components.analysis_runner import AnalysisPayload, _build_config
    from trend_analysis.config import ConfigType

    payload = AnalysisPayload(
        returns=returns,
        model_state=model_state,
        benchmark=benchmark,
    )
    config = _build_config(payload)
    return run_simulation(cast(ConfigType, config), _prepare_returns(returns))


def _select_metric_row(metrics: pd.DataFrame) -> pd.Series | None:
    if metrics.empty:
        return None
    for label in ("user_weight", "equal_weight"):
        if label in metrics.index:
            return metrics.loc[label]
    return metrics.iloc[0]


def _metric_value(metrics_row: pd.Series | None, keys: Iterable[str]) -> float:
    if metrics_row is None:
        return math.nan
    for key in keys:
        if key in metrics_row:
            value = metrics_row.get(key)
            if value is not None and not pd.isna(value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
    return math.nan


def _summarize_run(run_result: RunResult) -> dict[str, Any]:
    from trend_analysis.metrics import (
        annual_return,
        max_drawdown,
        sharpe_ratio,
        volatility,
    )

    metrics_row = _select_metric_row(run_result.metrics)
    sharpe = _metric_value(metrics_row, ("sharpe", "sharpe_ratio"))
    cagr = _metric_value(metrics_row, ("cagr", "CAGR", "annual_return"))
    vol = _metric_value(metrics_row, ("vol", "volatility"))
    max_dd = _metric_value(metrics_row, ("max_drawdown", "MaxDrawdown"))

    analysis = run_result.analysis
    if analysis is None:
        try:
            analysis = Results.from_payload(run_result.details)
        except Exception:
            analysis = None

    returns = analysis.returns if analysis is not None else pd.Series(dtype=float)
    weights = analysis.weights if analysis is not None else pd.Series(dtype=float)
    turnover = analysis.turnover if analysis is not None else pd.Series(dtype=float)
    costs = analysis.costs if analysis is not None else {}

    if not returns.empty:
        if math.isnan(cagr):
            cagr = float(annual_return(returns))
        if math.isnan(vol):
            vol = float(volatility(returns))
        if math.isnan(sharpe):
            sharpe = float(sharpe_ratio(returns))
        if math.isnan(max_dd):
            max_dd = float(max_drawdown(returns))

    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "volatility": vol,
        "max_drawdown": max_dd,
        "returns": returns,
        "weights": weights,
        "turnover": turnover,
        "costs": costs,
    }


def _weight_stats(a: pd.Series, b: pd.Series) -> dict[str, float]:
    if a.empty and b.empty:
        return {"l1": 0.0, "max_abs": 0.0, "active_change_count": 0.0}
    merged = pd.concat([a, b], axis=1).fillna(0.0)
    diff = (merged.iloc[:, 0] - merged.iloc[:, 1]).abs()
    return {
        "l1": float(diff.sum()),
        "max_abs": float(diff.max()),
        "active_change_count": float((diff > 1.0e-6).sum()),
    }


def _mean_turnover(turnover: pd.Series) -> float:
    if turnover is None or turnover.empty:
        return 0.0
    return float(turnover.abs().mean())


def _sign_flip_test(
    diff: pd.Series, *, seed: int = 42, iterations: int = 1000
) -> float:
    if diff.empty or len(diff) < 3:
        return math.nan
    rng = np.random.default_rng(seed)
    values = diff.values
    observed = float(values.mean())
    signs = rng.choice([-1, 1], size=(iterations, len(values)))
    means = (signs * values).mean(axis=1)
    p_value = float((np.abs(means) >= abs(observed)).mean())
    return p_value


def _total_return(returns: pd.Series) -> float:
    if returns is None or returns.empty:
        return math.nan
    return float((1.0 + returns).prod() - 1.0)


def _evaluate_setting(
    setting: str,
    baseline_state: dict[str, Any],
    returns: pd.DataFrame,
    cache: dict[str, Any],
    benchmark: str | None,
) -> SettingResult:
    baseline_value = baseline_state.get(setting)
    test_value, required_context = _resolve_variation(setting, baseline_value, returns)
    if test_value is None:
        return SettingResult(
            setting=setting,
            baseline_value=baseline_value,
            test_value=test_value,
            status="ERROR",
            mode_specific=bool(required_context),
            reason="No variation could be generated.",
            required_context=required_context,
        )

    baseline_with_context = _apply_context(baseline_state, required_context)
    test_state = dict(baseline_with_context)
    test_state[setting] = test_value

    if setting == "min_weight":
        max_weight = test_state.get("max_weight")
        if max_weight is not None and test_value >= max_weight:
            test_state[setting] = max(0.0, float(max_weight) * 0.5)
    if setting == "max_weight":
        min_weight = test_state.get("min_weight")
        if min_weight is not None and test_value <= min_weight:
            test_state[setting] = float(min_weight) * 1.5

    if setting in {"start_date", "end_date", "date_mode"}:
        test_state["date_mode"] = "explicit"
        test_state.setdefault("start_date", returns.index.min().strftime("%Y-%m-%d"))
        test_state.setdefault("end_date", returns.index.max().strftime("%Y-%m-%d"))

    try:
        base_key = json.dumps(baseline_with_context, sort_keys=True, default=str)
        if base_key in cache:
            base_result = cache[base_key]
        else:
            base_result = _run_simulation(returns, baseline_with_context, benchmark)
            cache[base_key] = base_result

        test_key = json.dumps(test_state, sort_keys=True, default=str)
        if test_key in cache:
            test_result = cache[test_key]
        else:
            test_result = _run_simulation(returns, test_state, benchmark)
            cache[test_key] = test_result
    except Exception as exc:
        return SettingResult(
            setting=setting,
            baseline_value=baseline_value,
            test_value=test_value,
            status="ERROR",
            mode_specific=bool(required_context),
            reason=f"Simulation failed: {exc}",
            required_context=required_context,
        )

    base_summary = _summarize_run(base_result)
    test_summary = _summarize_run(test_result)

    weight_stats = _weight_stats(base_summary["weights"], test_summary["weights"])
    weight_diff = weight_stats["l1"]
    weight_max_abs_diff = weight_stats["max_abs"]
    weight_change_count = weight_stats["active_change_count"]
    return_mean_diff = float(
        test_summary["returns"].mean() - base_summary["returns"].mean()
        if not base_summary["returns"].empty and not test_summary["returns"].empty
        else math.nan
    )
    mean_abs_return_diff = float(
        (test_summary["returns"] - base_summary["returns"]).abs().mean()
        if not base_summary["returns"].empty and not test_summary["returns"].empty
        else math.nan
    )
    total_return_diff = float(
        _total_return(test_summary["returns"]) - _total_return(base_summary["returns"])
    )
    return_vol_diff = float(
        test_summary["returns"].std() - base_summary["returns"].std()
        if not base_summary["returns"].empty and not test_summary["returns"].empty
        else math.nan
    )
    sharpe_diff = float(test_summary["sharpe"] - base_summary["sharpe"])
    cagr_diff = float(test_summary["cagr"] - base_summary["cagr"])
    vol_diff = float(test_summary["volatility"] - base_summary["volatility"])
    max_dd_diff = float(test_summary["max_drawdown"] - base_summary["max_drawdown"])
    turnover_diff = float(
        _mean_turnover(test_summary["turnover"])
        - _mean_turnover(base_summary["turnover"])
    )
    cost_diff = float(
        test_summary.get("costs", {}).get("turnover_applied", 0.0)
        - base_summary.get("costs", {}).get("turnover_applied", 0.0)
    )

    returns_diff = test_summary["returns"] - base_summary["returns"]
    tracking_error = float(returns_diff.std()) if not returns_diff.empty else math.nan
    return_corr = float(
        test_summary["returns"].corr(base_summary["returns"])
        if not base_summary["returns"].empty and not test_summary["returns"].empty
        else math.nan
    )
    p_value = _sign_flip_test(returns_diff.dropna())

    metric_changed = any(
        (
            not math.isnan(weight_diff) and abs(weight_diff) > 1.0e-4,
            not math.isnan(weight_max_abs_diff) and abs(weight_max_abs_diff) > 1.0e-4,
            not math.isnan(weight_change_count) and weight_change_count > 0,
            not math.isnan(return_mean_diff) and abs(return_mean_diff) > 1.0e-4,
            not math.isnan(mean_abs_return_diff) and abs(mean_abs_return_diff) > 1.0e-4,
            not math.isnan(total_return_diff) and abs(total_return_diff) > 1.0e-4,
            not math.isnan(return_vol_diff) and abs(return_vol_diff) > 1.0e-4,
            not math.isnan(sharpe_diff) and abs(sharpe_diff) > 1.0e-3,
            not math.isnan(cagr_diff) and abs(cagr_diff) > 1.0e-4,
            not math.isnan(vol_diff) and abs(vol_diff) > 1.0e-4,
            not math.isnan(max_dd_diff) and abs(max_dd_diff) > 1.0e-4,
            not math.isnan(turnover_diff) and abs(turnover_diff) > 1.0e-4,
        )
    )

    significant = (
        not math.isnan(p_value)
        and p_value < 0.05
        and not math.isnan(return_mean_diff)
        and abs(return_mean_diff) > 1.0e-4
    )

    reporting_only = setting.startswith(REPORTING_ONLY_PREFIXES) or (
        setting in REPORTING_ONLY_KEYS
    )

    if reporting_only:
        status = "NO_EFFECT"
        reason = "Reporting-only setting."
    elif metric_changed and (significant or math.isnan(p_value)):
        status = "MODE_SPECIFIC" if required_context else "EFFECTIVE"
        reason = "Detected meaningful changes."
    else:
        status = "NO_EFFECT"
        reason = "No meaningful changes detected."

    return SettingResult(
        setting=setting,
        baseline_value=baseline_value,
        test_value=test_value,
        status=status,
        mode_specific=bool(required_context),
        metrics={
            "weight_l1_diff": weight_diff,
            "weight_max_abs_diff": weight_max_abs_diff,
            "weight_change_count": weight_change_count,
            "mean_return_diff": return_mean_diff,
            "mean_abs_return_diff": mean_abs_return_diff,
            "total_return_diff": total_return_diff,
            "return_vol_diff": return_vol_diff,
            "tracking_error": tracking_error,
            "return_corr": return_corr,
            "sharpe_diff": sharpe_diff,
            "cagr_diff": cagr_diff,
            "volatility_diff": vol_diff,
            "max_drawdown_diff": max_dd_diff,
            "turnover_diff": turnover_diff,
            "cost_diff": cost_diff,
            "p_value": p_value,
            "significant": significant,
        },
        reason=reason,
        required_context=required_context,
    )


def _load_demo_data() -> pd.DataFrame:
    demo_path = PROJECT_ROOT / "demo" / "demo_returns.csv"
    if not demo_path.exists():
        from scripts.generate_demo import main as generate_demo

        generate_demo()
    df = pd.read_csv(demo_path, parse_dates=["Date"])
    return df.set_index("Date")


def _write_outputs(
    results: list[SettingResult],
    output_json: Path,
    output_csv: Path,
    settings: list[str],
) -> None:
    category_map = _load_setting_categories(TEST_WIRING_FILE)
    status_counts: dict[str, int] = {}
    category_stats: dict[str, dict[str, int]] = {}
    for res in results:
        status_counts[res.status] = status_counts.get(res.status, 0) + 1
        category = category_map.get(res.setting, "Uncategorized")
        if category not in category_stats:
            category_stats[category] = {"total": 0, "effective": 0}
        category_stats[category]["total"] += 1
        if res.status in ("EFFECTIVE", "MODE_SPECIFIC"):
            category_stats[category]["effective"] += 1
    total = len(results)
    effective = status_counts.get("EFFECTIVE", 0) + status_counts.get(
        "MODE_SPECIFIC", 0
    )
    effectiveness_rate = effective / total if total else 0.0

    by_category = {
        category: {
            "total": stats["total"],
            "effective": stats["effective"],
            "rate": stats["effective"] / stats["total"] if stats["total"] else 0.0,
        }
        for category, stats in sorted(category_stats.items())
    }

    non_effective = []
    for res in results:
        if res.status != "NO_EFFECT":
            continue
        non_effective.append(
            {
                "setting": res.setting,
                "category": category_map.get(res.setting, "Uncategorized"),
                "status": res.status,
                "reason": res.reason,
                "recommendation": _recommendation_for_result(res),
                "required_context": res.required_context,
            }
        )

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_settings": total,
        "status_counts": status_counts,
        "effectiveness_rate": effectiveness_rate,
        "by_category": by_category,
        "non_effective_settings": non_effective,
        "settings": [
            {
                "setting": res.setting,
                "baseline_value": res.baseline_value,
                "test_value": res.test_value,
                "status": res.status,
                "mode_specific": res.mode_specific,
                "metrics": res.metrics,
                "reason": res.reason,
                "required_context": res.required_context,
            }
            for res in results
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    rows = []
    for res in results:
        category = category_map.get(res.setting, "Uncategorized")
        row = {
            "setting": res.setting,
            "category": category,
            "status": res.status,
            "mode_specific": res.mode_specific,
            "baseline_value": json.dumps(res.baseline_value, default=str),
            "test_value": json.dumps(res.test_value, default=str),
            "reason": res.reason,
            "recommendation": _recommendation_for_result(res),
            "required_context": json.dumps(res.required_context, default=str),
        }
        row.update(res.metrics)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        default=str(PROJECT_ROOT / "reports" / "settings_effectiveness.json"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(PROJECT_ROOT / "reports" / "settings_effectiveness.csv"),
    )
    parser.add_argument("--benchmark", default=None)
    args = parser.parse_args()

    settings = sorted(_extract_settings_from_model(MODEL_FILE))
    baseline_state = _build_baseline_state(settings)
    returns = _load_demo_data()

    results: list[SettingResult] = []
    cache: dict[str, Any] = {}
    for setting in settings:
        result = _evaluate_setting(
            setting, baseline_state, returns, cache, args.benchmark
        )
        results.append(result)

    _write_outputs(
        results,
        Path(args.output_json),
        Path(args.output_csv),
        settings,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
