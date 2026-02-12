"""Helpers for extracting visualization artifact bundles."""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _with_counter_suffix(relative_path: Path, counter: int) -> Path:
    stem = relative_path.stem
    suffix = relative_path.suffix
    return relative_path.with_name(f"{stem}-{counter}{suffix}")


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
    written_rel_paths: set[str] = set()

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

            target_relative = relative_path
            collision_counter = 0
            while True:
                rel_key = target_relative.as_posix()
                target_path = destination_dir / target_relative
                if rel_key not in written_rel_paths and not target_path.exists():
                    break
                collision_counter += 1
                target_relative = _with_counter_suffix(relative_path, collision_counter)

            if collision_counter:
                message = (
                    f"Renamed extracted '{relative_path.as_posix()}' to "
                    f"'{target_relative.as_posix()}' to avoid collision."
                )
                warning_messages.append(message)
                logger.warning(message)

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_file, target_path.open("wb") as target_file:
                shutil.copyfileobj(source_file, target_file)
            written_rel_paths.add(target_relative.as_posix())

    return warning_messages
