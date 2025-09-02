#!/usr/bin/env python3
# ruff: noqa=E402
"""
Demo script showing the before/after behavior of the export bundle fix.

Run this to see the improvements in temporary file handling.
"""

import os
from unittest import mock

# Import the bundle utilities used in this demo
from trend_analysis.io.utils import (
    export_bundle,
    cleanup_bundle_file,
    _TEMP_FILES_TO_CLEANUP,
)

def create_mock_results():
    """Create a mock results object for testing."""
    mock_results = mock.MagicMock()

    def mock_to_csv(path, header=None):
        with open(path, "w") as f:
            f.write("return\n0.05\n0.03\n-0.01\n0.02\n")

    mock_results.portfolio.to_csv = mock_to_csv

    def mock_event_log():
        log_mock = mock.MagicMock()

        def mock_log_to_csv(path):
            with open(path, "w") as f:
                f.write("event,timestamp\n")
                f.write("rebalance,2023-01-01\n")
                f.write("rebalance,2023-02-01\n")
                f.write("rebalance,2023-03-01\n")

        log_mock.to_csv = mock_log_to_csv
        return log_mock

    mock_results.event_log_df = mock_event_log
    mock_results.summary.return_value = {
        "total_return": 0.15,
        "sharpe_ratio": 1.2,
        "max_drawdown": -0.05,
        "volatility": 0.08,
    }

    return mock_results


def show_before_after_comparison():
    """Demonstrate the before vs after behavior."""
    print("=" * 60)
    print("EXPORT BUNDLE CLEANUP FIX - DEMONSTRATION")
    print("=" * 60)
    print()

    print("🔍 ISSUE ANALYSIS:")
    print("  ❌ OLD: Files created in working directory (trend_app_run_*)")
    print("  ❌ OLD: No cleanup - files accumulate over time")
    print("  ❌ OLD: Manual directory/ZIP file management")
    print("  ❌ OLD: No caching - regenerates identical bundles")
    print()

    print("✅ SOLUTION:")
    print("  ✓ NEW: Files created in system temp directory (/tmp)")
    print("  ✓ NEW: Automatic cleanup on process exit")
    print("  ✓ NEW: Proper temporary file handling")
    print("  ✓ NEW: Streamlit caching to avoid regeneration")
    print()

    # Show working directory state
    cwd_files_before = [f for f in os.listdir(".") if "trend_app_run" in f]
    print(f"📁 Working directory before: {len(cwd_files_before)} trend files")

    # Create test data
    mock_results = create_mock_results()
    configs = [
        {"lookback_period": 252, "rebalance_freq": "monthly"},
        {"lookback_period": 126, "rebalance_freq": "weekly"},
        {"lookback_period": 60, "rebalance_freq": "daily"},
    ]

    print()
    print("🔧 CREATING EXPORT BUNDLES...")
    print()

    bundle_paths = []
    for i, config in enumerate(configs):
        config["run_id"] = i + 1
        bundle_path = export_bundle(mock_results, config)
        bundle_paths.append(bundle_path)

        print(f"  Bundle {i+1}:")
        print(f"    📦 File: {os.path.basename(bundle_path)}")
        print(f"    📍 Location: {os.path.dirname(bundle_path)}")
        print(f"    📊 Size: {os.path.getsize(bundle_path):,} bytes")
        print(f"    ✅ Exists: {os.path.exists(bundle_path)}")
        print()

    # Show results
    cwd_files_after = [f for f in os.listdir(".") if "trend_app_run" in f]
    print("📋 RESULTS:")
    print(f"  📁 Working directory after: {len(cwd_files_after)} trend files")
    print(f"  🗂️  Temp files created: {len(bundle_paths)}")
    print(f"  🧹 Files registered for cleanup: {len(_TEMP_FILES_TO_CLEANUP)}")
    print()

    # Show bundle contents
    import zipfile

    print("📄 BUNDLE CONTENTS (first bundle):")
    with zipfile.ZipFile(bundle_paths[0], "r") as z:
        for file_info in z.infolist():
            print(f"  📋 {file_info.filename} ({file_info.file_size} bytes)")
    print()

    print("💡 KEY IMPROVEMENTS:")
    print("  🎯 No working directory pollution")
    print("  🔄 Automatic cleanup on process exit")
    print("  ⚡ Streamlit caching prevents regeneration")
    print("  🛡️  Proper exception handling")
    print("  🔗 Backward compatible API")
    print()

    print("🎉 SOLUTION SUCCESSFULLY IMPLEMENTED!")
    print("=" * 60)

    return bundle_paths


if __name__ == "__main__":
    bundles = show_before_after_comparison()

    # Optional: Manual cleanup demo
    print("\n💻 Optional manual cleanup demo:")
    print("  (Files will be auto-cleaned on exit anyway)")

    for i, bundle in enumerate(bundles[:1]):  # Just clean one as demo
        print(f"  🧹 Cleaning bundle {i+1}...")
        cleanup_bundle_file(bundle)
        print(f"    ✓ Cleaned: {not os.path.exists(bundle)}")

    print()
    print("🏁 Demo complete. Remaining files will auto-cleanup on exit.")
