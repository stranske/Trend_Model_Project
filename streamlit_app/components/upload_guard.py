"""Helpers that enforce guardrails around uploaded files."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from streamlit.runtime.uploaded_file_manager import UploadedFile

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPLOAD_DIR = REPO_ROOT / "tmp" / "uploads"
ALLOWED_EXTENSIONS = frozenset({".csv", ".xlsx", ".xls"})
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB


class UploadViolation(ValueError):
    """Raised when an uploaded file violates guardrails."""


@dataclass(slots=True)
class GuardedUpload:
    """Metadata describing a validated upload."""

    original_name: str
    stored_path: Path
    data: bytes
    content_hash: str
    size: int

    @property
    def extension(self) -> str:
        return Path(self.original_name).suffix.lower()


def _normalise_extension(name: str) -> str:
    return Path(name).suffix.lower()


def _sanitize_stem(stem: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-")
    return safe or "upload"


def _format_size(num_bytes: int) -> str:
    mb = num_bytes / (1024**2)
    return f"{mb:.1f} MB"


def _ensure_upload_dir(base: Path | None) -> Path:
    target = (base or DEFAULT_UPLOAD_DIR).resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def hash_bytes(data: bytes) -> str:
    """Return a SHA256 hash for ``data``."""

    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def hash_path(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA256 hash for the file at ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def guard_and_buffer_upload(
    uploaded: UploadedFile,
    *,
    allowed_extensions: Iterable[str] = ALLOWED_EXTENSIONS,
    max_bytes: int = MAX_UPLOAD_BYTES,
    upload_dir: Path | None = None,
) -> GuardedUpload:
    """Validate ``uploaded`` and persist a safe copy to disk."""

    original_name = getattr(uploaded, "name", None)
    if not isinstance(original_name, str) or not original_name:
        raise UploadViolation("Uploaded file must have a filename.")

    extension = _normalise_extension(original_name)
    allowed_normalised = {ext.lower() for ext in allowed_extensions}
    if extension not in allowed_normalised:
        raise UploadViolation(
            "Unsupported file type. Please upload a CSV or Excel file."
        )

    uploaded.seek(0)
    data = uploaded.read()
    uploaded.seek(0)
    if not data:
        raise UploadViolation("Uploaded file is empty.")

    size = len(data)
    if size > max_bytes:
        raise UploadViolation(
            f"File too large: {_format_size(size)} (limit {_format_size(max_bytes)})."
        )

    content_hash = hash_bytes(data)
    upload_base = _ensure_upload_dir(upload_dir)
    stem = _sanitize_stem(Path(original_name).stem)
    safe_name = f"{stem}-{content_hash[:8]}{extension}"
    target = upload_base / safe_name

    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(upload_base))
    try:
        with os.fdopen(tmp_fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_path, target)
    except Exception:
        # Best-effort cleanup on failure
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    return GuardedUpload(
        original_name=original_name,
        stored_path=target,
        data=data,
        content_hash=content_hash,
        size=size,
    )
