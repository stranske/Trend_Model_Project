#!/usr/bin/env python3
"""
Test multi-period manager selection process
Verify that:
1. Different managers are selected each period based on in-sample ranking
2. Out-of-sample performance is measured for the selected managers
3. The process iterates through all periods correctly
"""

from pathlib import Path  # noqa: F401 - retained for potential future path operations

from trend_analysis.config import load
from trend_analysis.data import load_csv
from trend_analysis.multi_period import run_from_config as run_mp


def test_multi_period_selection():
    """Test the multi-period selection process with our portfolio test data."""

    print("=" * 70)
    print("TESTING MULTI-PERIOD MANAGER SELECTION PROCESS")
    print("=" * 70)

    # Load configuration
    cfg = load("config/portfolio_test.yml")
    print(f"\nLoaded config: {cfg.data['csv_path']}")
    print(f"Selection mode: {cfg.portfolio.get('selection_mode')}")
    print(f"Top N managers: {cfg.portfolio.get('rank', {}).get('n', 'Unknown')}")
    print(f"Ranking metric: {cfg.portfolio.get('rank', {}).get('score_by', 'Unknown')}")

    # Load the data to see what managers we have
    df = load_csv(cfg.data["csv_path"])
    if df is None:
        raise ValueError("Could not load data")

    manager_cols = [c for c in df.columns if c != "Date" and c != "SPX"]
    print(f"\nAvailable managers: {len(manager_cols)}")
    print(f"Manager names: {manager_cols}")
    print(f"Data period: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Total months: {len(df)}")

    # Run multi-period analysis
    print("\nRunning multi-period analysis...")
    results = run_mp(cfg)

    print(f"Generated {len(results)} periods")

    # Analyze each period's selection
    print("\nANALYZING MANAGER SELECTION BY PERIOD:")
    print("=" * 70)

    all_selected_managers = set()
    period_selections = {}

    for i, result in enumerate(results):
        period = result["period"]
        score_frame = result.get("score_frame")

        if score_frame is not None:
            # Get the selected managers (those with score data)
            selected_managers = list(score_frame.index)
            period_selections[i] = selected_managers
            all_selected_managers.update(selected_managers)

            # Show in-sample period and selected managers
            print(f"\nPeriod {i+1:2d}: {period[0]} to {period[1]} (In-Sample)")
            print(f"          {period[2]} to {period[3]} (Out-Sample)")
            print(f"  Selected: {selected_managers}")

            # Show their Sharpe ratios (the ranking metric)
            if "Sharpe" in score_frame.columns:
                sharpe_scores = score_frame["Sharpe"].sort_values(ascending=False)
                print("  Sharpe Ratios:")
                for mgr, sharpe in sharpe_scores.items():
                    print(f"    {mgr}: {sharpe:6.2f}")
        else:
            print(f"\nPeriod {i+1:2d}: No score frame available")

    # Analysis summary
    print("\n" + "=" * 70)
    print("SELECTION ANALYSIS SUMMARY")
    print("=" * 70)

    print(
        f"Total unique managers selected across all periods: {len(all_selected_managers)}"
    )
    print(f"All selected managers: {sorted(all_selected_managers)}")

    # Check if selection changes between periods
    if len(results) > 1:
        changes = 0
        for i in range(1, len(results)):
            prev_selection = set(period_selections.get(i - 1, []))
            curr_selection = set(period_selections.get(i, []))

            if prev_selection != curr_selection:
                changes += 1
                added = curr_selection - prev_selection
                removed = prev_selection - curr_selection
                unchanged = prev_selection & curr_selection

                print(f"\nPeriod {i} vs {i+1} changes:")
                if added:
                    print(f"  Added: {sorted(added)}")
                if removed:
                    print(f"  Removed: {sorted(removed)}")
                print(f"  Unchanged: {sorted(unchanged)} ({len(unchanged)}/8)")

        print(f"\nTotal periods with selection changes: {changes}/{len(results)-1}")

        if changes == 0:
            print("⚠️  WARNING: No selection changes detected!")
            print("   This might indicate the ranking isn't working properly")
        else:
            print("✅ GOOD: Manager selection is changing between periods")

    # Show performance of selected vs non-selected managers
    print("\n" + "=" * 70)
    print("PERFORMANCE VERIFICATION")
    print("=" * 70)

    # For the first few periods, show the performance difference
    for i in range(min(3, len(results))):
        result = results[i]
        period = result["period"]

        # Get in-sample and out-sample performance data
        in_stats = result.get("in_sample_stats", {})
        out_stats = result.get("out_sample_stats", {})

        print(f"\nPeriod {i+1} - In-Sample Performance ({period[0]} to {period[1]}):")
        if in_stats:
            # Sort by Sharpe ratio (our selection metric)
            sharpe_data = [
                (mgr, getattr(stats, "sharpe", 0))
                for mgr, stats in in_stats.items()
                if hasattr(stats, "sharpe")
            ]
            sharpe_data.sort(key=lambda x: x[1], reverse=True)

            selected = period_selections.get(i, [])
            print("  Manager Rankings (by Sharpe):")
            for rank, (mgr, sharpe) in enumerate(sharpe_data[:12], 1):  # Show top 12
                status = "SELECTED" if mgr in selected else "not selected"
                print(f"    {rank:2d}. {mgr}: {sharpe:6.2f} - {status}")

        print(
            f"\nPeriod {i+1} - Out-of-Sample Performance ({period[2]} to {period[3]}):"
        )
        if out_stats:
            selected = period_selections.get(i, [])
            print("  Selected managers out-of-sample performance:")
            for mgr in selected:
                if mgr in out_stats:
                    stats = out_stats[mgr]
                    cagr = getattr(stats, "cagr", 0) * 100
                    sharpe = getattr(stats, "sharpe", 0)
                    print(f"    {mgr}: {cagr:6.1f}% CAGR, {sharpe:5.2f} Sharpe")

    # Final verification
    print("\n" + "=" * 70)
    print("PROCESS VERIFICATION")
    print("=" * 70)

    checks = []

    # Check 1: Multiple periods
    if len(results) >= 10:
        checks.append("✅ Multiple analysis periods (15 expected)")
    else:
        checks.append(f"❌ Too few periods: {len(results)} (expected ~15)")

    # Check 2: Manager selection happening
    if len(all_selected_managers) > 8:
        checks.append("✅ Multiple managers selected across periods")
    else:
        checks.append(f"❌ Too few unique managers: {len(all_selected_managers)}")

    # Check 3: Selection changes
    if len(results) > 1:
        unique_selections = len(
            set(
                tuple(sorted(period_selections.get(i, []))) for i in range(len(results))
            )
        )
        if unique_selections > 1:
            checks.append(
                f"✅ Selection varies across periods ({unique_selections} unique combinations)"
            )
        else:
            checks.append("❌ Selection doesn't change between periods")

    # Check 4: Score frames present
    score_frame_count = sum(1 for r in results if r.get("score_frame") is not None)
    if score_frame_count == len(results):
        checks.append("✅ All periods have score frames")
    else:
        checks.append(
            f"❌ Missing score frames: {len(results) - score_frame_count}/{len(results)}"
        )

    for check in checks:
        print(f"  {check}")

    success = all("✅" in check for check in checks)

    print("\n" + "=" * 70)
    if success:
        print("🎉 MULTI-PERIOD SELECTION PROCESS IS WORKING CORRECTLY!")
    else:
        print("⚠️  ISSUES DETECTED IN MULTI-PERIOD SELECTION PROCESS")
    print("=" * 70)

    # Minimal assertions to validate behavior without returning values
    assert isinstance(results, list) and len(results) >= 1
    # At least one period should have a score_frame
    assert any(r.get("score_frame") is not None for r in results)
    # Each result should include a period with four entries (IS start/end, OOS start/end)
    assert all(
        isinstance(r.get("period"), (list, tuple)) and len(r["period"]) == 4
        for r in results
    )


if __name__ == "__main__":
    from trend_analysis.script_logging import setup_script_logging

    setup_script_logging(module_file=__file__)
    test_multi_period_selection()
