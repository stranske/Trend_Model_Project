#!/usr/bin/env python3
"""Generate evidence logs for each UI setting.

Uses the existing codebase's data loading and analysis infrastructure
to test each setting one at a time with real Trend Universe data.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.components.data_schema import load_and_validate_file  # noqa: E402
from scripts.test_settings_wiring import (  # noqa: E402
    SETTINGS_TO_TEST,
    get_baseline_state,
    run_analysis_with_state,
    extract_metric,
)

# Evidence output directory
EVIDENCE_DIR = PROJECT_ROOT / "docs" / "settings_evidence"


def load_trend_universe_data() -> pd.DataFrame:
    """Load the real Trend Universe data using existing infrastructure."""
    data_path = PROJECT_ROOT / "data" / "Trend Universe Data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Trend Universe data not found at {data_path}")

    # Use the same loading function as the Streamlit app
    # This handles date parsing, missing data policy, etc.
    df, meta = load_and_validate_file(data_path)
    print(f"Loaded Trend Universe data: {len(df)} rows × {len(df.columns)} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Missing policy: {meta.get('missing_policy', 'unknown')}")
    return df


def format_value(value: Any) -> str:
    """Format a value for display in evidence logs."""
    if value is None:
        return "None"
    if isinstance(value, float):
        if abs(value) < 0.0001:
            return f"{value:.6f}"
        return f"{value:.4f}"
    if isinstance(value, pd.Series):
        return f"Series({len(value)} items)"
    if isinstance(value, pd.DataFrame):
        return f"DataFrame({value.shape[0]}×{value.shape[1]})"
    return str(value)


def run_single_setting_test(
    setting_name: str,
    setting_config: dict[str, Any],
    returns: pd.DataFrame,
    baseline_state: dict[str, Any],
) -> dict[str, Any]:
    """Run a single setting test and return detailed evidence."""
    evidence: dict[str, Any] = {
        "setting_name": setting_name,
        "timestamp": datetime.now().isoformat(),
        "status": "UNKNOWN",
        "baseline_value": None,
        "test_value": None,
        "metric_name": setting_config.get("expected_metric", "unknown"),
        "expected_direction": setting_config.get("expected_direction", "change"),
    }

    # Build test state
    test_state = baseline_state.copy()
    test_state[setting_name] = setting_config["test_value"]

    # Apply any additional state changes (e.g., for inclusion_approach)
    if "additional_state" in setting_config:
        test_state.update(setting_config["additional_state"])

    evidence["baseline_state_value"] = baseline_state.get(setting_name)
    evidence["test_state_value"] = test_state[setting_name]

    # Run baseline analysis
    try:
        baseline_result = run_analysis_with_state(returns, baseline_state)
        baseline_metric = extract_metric(
            baseline_result,
            setting_config["expected_metric"],
            baseline_state,
        )
        evidence["baseline_value"] = baseline_metric
        evidence["baseline_formatted"] = format_value(baseline_metric)
    except Exception as e:
        evidence["status"] = "ERROR"
        evidence["error"] = f"Baseline analysis failed: {e}"
        return evidence

    # Run test analysis
    try:
        test_result = run_analysis_with_state(returns, test_state)
        test_metric = extract_metric(
            test_result,
            setting_config["expected_metric"],
            test_state,
        )
        evidence["test_value"] = test_metric
        evidence["test_formatted"] = format_value(test_metric)
    except Exception as e:
        evidence["status"] = "ERROR"
        evidence["error"] = f"Test analysis failed: {e}"
        return evidence

    # Check if metric changed
    baseline_val = evidence["baseline_value"]
    test_val = evidence["test_value"]

    if baseline_val is None or test_val is None:
        evidence["status"] = "SKIP"
        evidence["reason"] = "Could not extract metric values"
        return evidence

    # Determine if change occurred and direction
    try:
        baseline_num = float(baseline_val)
        test_num = float(test_val)
        changed = abs(baseline_num - test_num) > 1e-9
        evidence["metric_changed"] = changed

        if changed:
            if test_num > baseline_num:
                actual_direction = "increase"
            else:
                actual_direction = "decrease"
            evidence["actual_direction"] = actual_direction

            expected = setting_config.get("expected_direction", "change")
            if expected in ("change", "any"):
                evidence["direction_correct"] = True
            else:
                evidence["direction_correct"] = actual_direction == expected
        else:
            evidence["direction_correct"] = False

    except (TypeError, ValueError):
        # Non-numeric comparison
        changed = baseline_val != test_val
        evidence["metric_changed"] = changed
        evidence["direction_correct"] = changed

    # Determine final status
    if not evidence["metric_changed"]:
        evidence["status"] = "FAIL"
        evidence["reason"] = "Setting had no effect on output"
    elif evidence.get("direction_correct", False):
        evidence["status"] = "PASS"
    else:
        evidence["status"] = "WARN"
        evidence["reason"] = (
            f"Changed in wrong direction: expected {setting_config.get('expected_direction')}, "
            f"got {evidence.get('actual_direction', 'unknown')}"
        )

    return evidence


def generate_evidence_markdown(evidence: dict[str, Any]) -> str:
    """Generate a markdown report for a single setting test."""
    lines = [
        f"# Setting: `{evidence['setting_name']}`",
        "",
        f"**Test Date:** {evidence['timestamp']}",
        f"**Status:** {evidence['status']}",
        "",
        "## Configuration",
        "",
        f"- **Baseline Value:** `{evidence.get('baseline_state_value')}`",
        f"- **Test Value:** `{evidence.get('test_state_value')}`",
        f"- **Expected Metric:** `{evidence['metric_name']}`",
        f"- **Expected Direction:** `{evidence['expected_direction']}`",
        "",
        "## Results",
        "",
        f"- **Baseline Metric:** {evidence.get('baseline_formatted', 'N/A')}",
        f"- **Test Metric:** {evidence.get('test_formatted', 'N/A')}",
        f"- **Metric Changed:** {evidence.get('metric_changed', 'N/A')}",
    ]

    if "actual_direction" in evidence:
        lines.append(f"- **Actual Direction:** {evidence['actual_direction']}")

    if "direction_correct" in evidence:
        lines.append(f"- **Direction Correct:** {evidence['direction_correct']}")

    if "reason" in evidence:
        lines.extend(["", f"**Note:** {evidence['reason']}"])

    if "error" in evidence:
        lines.extend(["", f"**Error:** {evidence['error']}"])

    lines.extend(
        [
            "",
            "## Economic Interpretation",
            "",
        ]
    )

    # Add economic interpretation based on the setting
    interpretation = get_economic_interpretation(evidence)
    lines.append(interpretation)

    return "\n".join(lines)


def get_economic_interpretation(evidence: dict[str, Any]) -> str:
    """Generate economic interpretation for a setting test result."""
    setting = evidence["setting_name"]
    status = evidence["status"]
    baseline = evidence.get("baseline_formatted", "N/A")
    test = evidence.get("test_formatted", "N/A")
    direction = evidence.get("actual_direction", "unchanged")

    interpretations = {
        "selection_count": (
            f"The number of funds in the portfolio changed from baseline to test. "
            f"With baseline={baseline} and test={test}, the portfolio "
            f"{'became more concentrated' if direction == 'decrease' else 'became more diversified'}. "
            f"Economic intuition: More funds typically reduce idiosyncratic risk but may dilute alpha."
        ),
        "max_weight": (
            f"Maximum position size constraint. Baseline={baseline}, Test={test}. "
            f"{'Tighter constraints' if direction == 'decrease' else 'Looser constraints'} on concentration. "
            f"Economic intuition: Lower max weights force diversification, potentially reducing volatility."
        ),
        "lookback_periods": (
            f"In-sample evaluation window changed. Baseline={baseline}, Test={test}. "
            f"{'Shorter' if direction == 'decrease' else 'Longer'} lookback for ranking funds. "
            f"Economic intuition: Longer lookbacks capture more trend persistence but may miss regime changes."
        ),
        "trend_window": (
            f"Trend signal calculation window. Baseline={baseline}, Test={test}. "
            f"{'Shorter' if direction == 'decrease' else 'Longer'} window for momentum. "
            f"Economic intuition: Shorter windows capture faster trends, longer windows reduce noise."
        ),
        "risk_target": (
            f"Portfolio volatility target. Baseline={baseline}, Test={test}. "
            f"{'Lower' if direction == 'decrease' else 'Higher'} risk appetite. "
            f"Economic intuition: Higher risk targets allow more aggressive positioning."
        ),
    }

    if setting in interpretations:
        return interpretations[setting]

    # Generic interpretation
    if direction == "increase":
        direction_phrase = "increased"
    elif direction == "decrease":
        direction_phrase = "decreased"
    elif direction == "unchanged":
        direction_phrase = "remained unchanged"
    else:
        # Fallback: use the raw direction string if it's already a phrase
        direction_phrase = direction

    return (
        f"Setting `{setting}` was tested. Baseline metric: {baseline}, Test metric: {test}. "
        f"The metric {direction_phrase} as expected."
        if status == "PASS"
        else f"The setting change {'had the expected effect' if status == 'PASS' else 'requires further investigation'}."
    )


def generate_summary_report(all_evidence: list[dict[str, Any]]) -> str:
    """Generate a summary markdown report."""
    passed = [e for e in all_evidence if e["status"] == "PASS"]
    failed = [e for e in all_evidence if e["status"] == "FAIL"]
    warned = [e for e in all_evidence if e["status"] == "WARN"]
    errored = [e for e in all_evidence if e["status"] == "ERROR"]
    skipped = [e for e in all_evidence if e["status"] == "SKIP"]

    lines = [
        "# Settings Wiring Evidence Summary",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Data Source:** Trend Universe Data",
        "",
        "## Overview",
        "",
        f"| Status | Count |",
        f"|--------|-------|",
        f"| ✅ PASS | {len(passed)} |",
        f"| ❌ FAIL | {len(failed)} |",
        f"| ⚠️ WARN | {len(warned)} |",
        f"| 🚫 ERROR | {len(errored)} |",
        f"| ⏭️ SKIP | {len(skipped)} |",
        f"| **Total** | **{len(all_evidence)}** |",
        "",
    ]

    if passed:
        lines.extend(
            [
                "## ✅ Passing Settings",
                "",
                "| Setting | Baseline | Test | Direction |",
                "|---------|----------|------|-----------|",
            ]
        )
        for e in passed:
            lines.append(
                f"| `{e['setting_name']}` | {e.get('baseline_formatted', 'N/A')} | "
                f"{e.get('test_formatted', 'N/A')} | {e.get('actual_direction', '-')} |"
            )
        lines.append("")

    if failed:
        lines.extend(
            [
                "## ❌ Failing Settings",
                "",
                "| Setting | Reason |",
                "|---------|--------|",
            ]
        )
        for e in failed:
            lines.append(
                f"| `{e['setting_name']}` | {e.get('reason', 'No change detected')} |"
            )
        lines.append("")

    if warned:
        lines.extend(
            [
                "## ⚠️ Warnings",
                "",
                "| Setting | Issue |",
                "|---------|-------|",
            ]
        )
        for e in warned:
            lines.append(
                f"| `{e['setting_name']}` | {e.get('reason', 'Unexpected direction')} |"
            )
        lines.append("")

    if errored:
        lines.extend(
            [
                "## 🚫 Errors",
                "",
                "| Setting | Error |",
                "|---------|-------|",
            ]
        )
        for e in errored:
            lines.append(
                f"| `{e['setting_name']}` | {e.get('error', 'Unknown error')} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate settings wiring evidence")
    parser.add_argument(
        "--setting",
        "-s",
        type=str,
        default=None,
        help="Test only this specific setting",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available settings",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    # List settings if requested
    if args.list:
        print("Available settings to test:")
        for setting in SETTINGS_TO_TEST:
            print(f"  - {setting.name}")
        return 0

    # Create evidence directory
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading Trend Universe data...")
    try:
        returns = load_trend_universe_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    # Get baseline state
    baseline_state = get_baseline_state()
    print(f"Baseline state has {len(baseline_state)} settings")

    # Determine which settings to test
    if args.setting:
        settings_to_run = [s for s in SETTINGS_TO_TEST if s.name == args.setting]
        if not settings_to_run:
            print(f"Setting '{args.setting}' not found")
            return 1
    else:
        settings_to_run = SETTINGS_TO_TEST

    print(f"\nRunning {len(settings_to_run)} setting tests...\n")

    all_evidence: list[dict[str, Any]] = []

    for i, setting in enumerate(settings_to_run, 1):
        print(f"[{i}/{len(settings_to_run)}] Testing: {setting.name}...", end=" ")

        setting_config = {
            "test_value": setting.test_value,
            "expected_metric": setting.expected_metric,
            "expected_direction": setting.expected_direction,
        }

        evidence = run_single_setting_test(
            setting.name,
            setting_config,
            returns,
            baseline_state,
        )

        all_evidence.append(evidence)

        status_icons = {
            "PASS": "✅",
            "FAIL": "❌",
            "WARN": "⚠️",
            "ERROR": "🚫",
            "SKIP": "⏭️",
        }
        print(f"{status_icons.get(evidence['status'], '?')} {evidence['status']}")

        if args.verbose and evidence.get("reason"):
            print(f"     → {evidence['reason']}")

        # Write individual evidence file
        evidence_file = EVIDENCE_DIR / f"{setting.name}.md"
        evidence_file.write_text(generate_evidence_markdown(evidence))

        # Also write JSON for programmatic access
        json_file = EVIDENCE_DIR / f"{setting.name}.json"
        # Convert non-serializable values
        json_safe = {
            k: (
                str(v)
                if not isinstance(v, (str, int, float, bool, type(None), list, dict))
                else v
            )
            for k, v in evidence.items()
        }
        json_file.write_text(json.dumps(json_safe, indent=2))

    # Generate summary report
    summary_file = EVIDENCE_DIR / "SUMMARY.md"
    summary_file.write_text(generate_summary_report(all_evidence))
    print(f"\n📊 Summary written to: {summary_file}")

    # Print final summary
    passed = sum(1 for e in all_evidence if e["status"] == "PASS")
    failed = sum(1 for e in all_evidence if e["status"] == "FAIL")
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(all_evidence)} tests")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
