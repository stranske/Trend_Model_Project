"""Input/Output utilities for trend analysis."""

from __future__ import annotations

import atexit
import datetime
import json
import os
import tempfile
import zipfile
from typing import Any

# Global registry for cleanup of temporary files
_TEMP_FILES_TO_CLEANUP: list[str] = []


def _cleanup_temp_files() -> None:
    """Clean up all registered temporary files on exit."""
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


def export_bundle(results: Any, config_dict: dict[str, Any]) -> str:
    """Export analysis results as a ZIP bundle.

    The bundle is assembled in-memory to avoid leaving intermediate files on
    disk.  A temporary ZIP file is created and registered for cleanup on exit.

    Returns:
        Path to the created ZIP file. The file will be automatically cleaned up
        on process exit, or can be manually cleaned up using cleanup_bundle_file().
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create temporary directory for bundle contents
    with tempfile.TemporaryDirectory(prefix=f"trend_app_run_{ts}_") as temp_dir:
        # Write bundle files to temporary directory
        # Write portfolio_returns.csv with exception handling
        try:
            results.portfolio.to_csv(
                os.path.join(temp_dir, "portfolio_returns.csv"), header=["return"]
            )
        except Exception:
            # Write empty CSV if export fails
            with open(os.path.join(temp_dir, "portfolio_returns.csv"), "w", encoding="utf-8") as f:
                f.write("return\n")

        # Write event_log.csv with exception handling
        ev = results.event_log_df()
        try:
            ev.to_csv(os.path.join(temp_dir, "event_log.csv"))
        except Exception:
            # Write empty CSV if export fails
            with open(os.path.join(temp_dir, "event_log.csv"), "w", encoding="utf-8") as f:
                f.write("")
        with open(os.path.join(temp_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(results.summary(), f, indent=2)
        with open(os.path.join(temp_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, default=str)

        # Create temporary ZIP file
        fd, zip_path = tempfile.mkstemp(suffix=f"_trend_bundle_{ts}.zip")
        os.close(fd)
        # Write ZIP content
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for root, _, files in os.walk(temp_dir):
                    for name in files:
                        p = os.path.join(root, name)
                        z.write(p, os.path.relpath(p, temp_dir))
        except Exception:
            # Clean up zip file if creation failed
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise

    _TEMP_FILES_TO_CLEANUP.append(zip_path)
    return zip_path


def cleanup_bundle_file(file_path: str) -> None:
    """Manually clean up a bundle file created by export_bundle.

    Args:
        file_path: The path to the bundle file returned by export_bundle.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        if file_path in _TEMP_FILES_TO_CLEANUP:
            _TEMP_FILES_TO_CLEANUP.remove(file_path)
    except Exception:
        # Silently ignore cleanup errors
        pass
