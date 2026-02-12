"""Helpers for extracting visualization artifact bundles."""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_bundle_zip(
    bundle_path: Path,
    destination_dir: Path,
    *,
    warnings: list[str] | None = None,
) -> list[str]:
    """Extract a chart bundle ZIP into ``destination_dir`` without overwrites.

    Parameters:
    - bundle_path: ZIP archive produced by chart export.
    - destination_dir: Directory where extracted files should be written.
    - warnings: Optional mutable list to append warning strings into.

    Returns:
    - A list containing warning messages produced during extraction.
    """

    warning_messages = warnings if warnings is not None else []
    destination_dir.mkdir(parents=True, exist_ok=True)
    seen_members: set[str] = set()

    with zipfile.ZipFile(bundle_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            relative_path = Path(member.filename)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                message = (
                    f"Skipped extracting '{member.filename}' because it resolves outside "
                    "the destination directory."
                )
                warning_messages.append(message)
                logger.warning(message)
                continue

            rel_key = relative_path.as_posix()
            if rel_key in seen_members:
                message = (
                    f"Skipped extracting duplicate entry '{rel_key}' because the filename "
                    "already appeared in the archive."
                )
                warning_messages.append(message)
                logger.warning(message)
                continue
            seen_members.add(rel_key)

            target_path = destination_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists():
                message = (
                    f"Skipped extracting '{rel_key}' because the destination file already exists."
                )
                warning_messages.append(message)
                logger.warning(message)
                continue

            with archive.open(member) as source_file, target_path.open("wb") as target_file:
                shutil.copyfileobj(source_file, target_file)

    return warning_messages
