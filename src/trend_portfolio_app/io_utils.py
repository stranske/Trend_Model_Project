from __future__ import annotations
import os
import json
import datetime
import zipfile
import tempfile
import shutil
import atexit
from typing import Tuple


# Global registry for cleanup of temporary files
_TEMP_FILES_TO_CLEANUP: list[str] = []


def _cleanup_temp_files() -> None:
    """Clean up all registered temporary files on exit."""
    global _TEMP_FILES_TO_CLEANUP
    for file_path in _TEMP_FILES_TO_CLEANUP:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            # Silently ignore cleanup errors to avoid exit issues
            pass
    _TEMP_FILES_TO_CLEANUP.clear()


# Register cleanup function to run on exit
atexit.register(_cleanup_temp_files)


def export_bundle(results, config_dict) -> str:
    """
    Export analysis results as a ZIP bundle using temporary files.

    Returns:
        Path to the created ZIP file. The file will be automatically cleaned up
        on process exit, or can be manually cleaned up using cleanup_bundle_file().
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create temporary directory for bundle contents
    with tempfile.TemporaryDirectory(prefix=f"trend_app_run_{ts}_") as temp_dir:
        # Write bundle files to temporary directory
        results.portfolio.to_csv(
            os.path.join(temp_dir, "portfolio_returns.csv"), header=["return"]
        )
        ev = results.event_log_df()
        ev.to_csv(os.path.join(temp_dir, "event_log.csv"))
        with open(os.path.join(temp_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(results.summary(), f, indent=2)
        with open(os.path.join(temp_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, default=str)

        # Create temporary ZIP file
        zip_fd, zip_path = tempfile.mkstemp(suffix=f"_trend_bundle_{ts}.zip")
        try:
            with os.fdopen(zip_fd, "wb") as zip_file:
                with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as z:
                    for root, _, files in os.walk(temp_dir):
                        for name in files:
                            p = os.path.join(root, name)
                            z.write(p, os.path.relpath(p, temp_dir))
        except Exception:
            # Clean up zip file if creation failed
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise

    # Register ZIP file for cleanup on exit
    _TEMP_FILES_TO_CLEANUP.append(zip_path)

    return zip_path


def cleanup_bundle_file(file_path: str) -> None:
    """
    Manually clean up a bundle file created by export_bundle.

    Args:
        file_path: The path to the bundle file returned by export_bundle.
    """
    global _TEMP_FILES_TO_CLEANUP
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        if file_path in _TEMP_FILES_TO_CLEANUP:
            _TEMP_FILES_TO_CLEANUP.remove(file_path)
    except Exception:
        # Silently ignore cleanup errors
        pass
