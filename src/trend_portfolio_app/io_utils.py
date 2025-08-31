from __future__ import annotations

import os
import json
import datetime
import zipfile
import tempfile
import atexit
import io


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
    Export analysis results as a ZIP bundle.

    The bundle is assembled in-memory to avoid leaving intermediate files on
    disk.  A temporary ZIP file is created and registered for cleanup on exit.

    Returns:
        Path to the created ZIP file. The file will be automatically cleaned up
        on process exit, or can be manually cleaned up using cleanup_bundle_file().
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    zip_fd, zip_path = tempfile.mkstemp(suffix=f"_trend_bundle_{ts}.zip")
    try:
        with os.fdopen(zip_fd, "wb") as zip_file:
            with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as z:
                try:
                    buf = io.StringIO()
                    results.portfolio.to_csv(buf, header=["return"])
                    z.writestr("portfolio_returns.csv", buf.getvalue())
                except Exception:
                    z.writestr("portfolio_returns.csv", "")

                try:
                    buf = io.StringIO()
                    results.event_log_df().to_csv(buf)
                    z.writestr("event_log.csv", buf.getvalue())
                except Exception:
                    z.writestr("event_log.csv", "")

                z.writestr("summary.json", json.dumps(results.summary(), indent=2))
                z.writestr(
                    "config.json",
                    json.dumps(config_dict, indent=2, default=str),
                )
    except Exception:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise

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
