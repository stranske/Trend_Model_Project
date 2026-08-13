#!/usr/bin/env python3
"""Regenerate the committed Pyodide runtime artifacts for the stlite demo.

The wheels are COMMITTED so the demo boots offline with zero network access;
this script regenerates or updates them from the Pyodide CDN when maintainers
intentionally refresh the vendored runtime. Pure-PyPI wheels outside the lock
are fetched separately from PyPI using pinned filenames.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from build_wasm_demo import (
    PYODIDE_LOCK_PYPI_OVERRIDES,
    PYPI_WHEEL_DIR,
    PYPI_WHEELS,
    REPO_ROOT,
    REQUIREMENTS,
)

PYODIDE_VERSION = "0.27.2"
CDN_BASE_URL = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full"
VENDOR_DIR = REPO_ROOT / "demo" / "wasm" / "vendor" / f"pyodide-{PYODIDE_VERSION}"
LOCK_PATH = VENDOR_DIR / "pyodide-lock.json"
DOWNLOAD_TIMEOUT_SECONDS = 30

#: Pure-PyPI packages outside the Pyodide lock are fetched from PyPI
#: (files.pythonhosted.org) by pinned filename and committed under
#: ``demo/wasm/vendor/pypi/`` for offline boot.
PYPI_VENDOR_DIR = REPO_ROOT / "demo" / "wasm" / "vendor" / PYPI_WHEEL_DIR
PYPI_JSON_URL = "https://pypi.org/pypi/{name}/{version}/json"
LANGCHAIN_CORE_WHEEL = "langchain_core-0.1.0-py3-none-any.whl"


def _package_name(requirement: str) -> str:
    return re.split(r"[<>=!~;,\[]", requirement, maxsplit=1)[0].strip()


def _lock_key(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _load_lock() -> dict[str, dict]:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))["packages"]


def _vendored_package_names(packages: dict[str, dict]) -> set[str]:
    vendored_files = {path.name for path in VENDOR_DIR.iterdir() if path.is_file()}
    return {
        name for name, package in packages.items() if package.get("file_name") in vendored_files
    }


def _seed_names(packages: dict[str, dict]) -> set[str]:
    manifest_requirement_names = {
        _lock_key(_package_name(requirement))
        for requirements in REQUIREMENTS.values()
        for requirement in requirements
    }
    pyodide_runtime_names = {"micropip", "packaging"}
    already_vendored_names = _vendored_package_names(packages)
    return {
        name
        for name in (manifest_requirement_names | pyodide_runtime_names | already_vendored_names)
        if name in packages
    }


def _dependency_closure(seed_names: set[str], packages: dict[str, dict]) -> list[str]:
    closure: set[str] = set()
    stack = list(seed_names)
    while stack:
        name = stack.pop()
        if name in closure:
            continue
        closure.add(name)
        stack.extend(_lock_key(dep) for dep in packages[name].get("depends", []))
    return sorted(closure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_package_file_name(package: dict[str, str]) -> str:
    file_name = package["file_name"]
    path = Path(file_name)
    if path.is_absolute() or path.name != file_name or ".." in path.parts or "\\" in file_name:
        raise ValueError(f"unsafe package file_name: {file_name!r}")
    return file_name


def _download(package: dict[str, str]) -> str:
    file_name = _safe_package_file_name(package)
    target = VENDOR_DIR / file_name
    expected_sha = package.get("sha256")
    if target.is_file():
        if expected_sha and _sha256(target) != expected_sha:
            target.unlink(missing_ok=True)
        else:
            return f"skip existing {file_name}"

    url = f"{CDN_BASE_URL}/{file_name}"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=VENDOR_DIR, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise

    if expected_sha and _sha256(tmp_path) != expected_sha:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {file_name}")

    tmp_path.replace(target)
    return f"downloaded {file_name}"


def _download_pypi_wheel(file_name: str, target_dir: Path) -> str:
    """Fetch one pinned PyPI wheel into ``target_dir`` (sha256-verified)."""

    target = target_dir / file_name
    rel = f"{target_dir.name}/{file_name}"
    if target.is_file():
        return f"skip existing {rel}"

    # Wheel filename is ``{name}-{version}-{pytag}-{abitag}-{plat}.whl``.
    name, version = file_name[: -len(".whl")].split("-")[:2]
    meta = json.loads(
        urllib.request.urlopen(
            PYPI_JSON_URL.format(name=name, version=version),
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        ).read()
    )
    wheel = next(item for item in meta["urls"] if item["filename"] == file_name)
    url = wheel["url"]
    if "files.pythonhosted.org" not in url:
        raise RuntimeError(f"unexpected wheel host for {file_name}: {url}")

    target_dir.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target_dir, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise

    expected_sha = wheel.get("digests", {}).get("sha256")
    if expected_sha and _sha256(tmp_path) != expected_sha:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {file_name}")

    tmp_path.replace(target)
    return f"downloaded {rel}"


def _patch_langchain_core_metadata(path: Path) -> str:
    """Relax two conservative bounds that conflict with the stlite bootstrap.

    LangChain core 0.1.0 runs with the lock's packaging 24.2 and stlite's
    tenacity 9.1.4, but its wheel metadata caps both versions. Pyodide refuses
    to downgrade packages that stlite already imported, so the committed wheel
    removes only those upper bounds and updates its RECORD hash.
    """

    metadata_name = "langchain_core-0.1.0.dist-info/METADATA"
    record_name = "langchain_core-0.1.0.dist-info/RECORD"
    with zipfile.ZipFile(path) as source:
        entries = [(info, source.read(info.filename)) for info in source.infolist()]
    payloads = {info.filename: payload for info, payload in entries}
    metadata = payloads[metadata_name].decode("utf-8")
    patched = metadata.replace(
        "Requires-Dist: packaging (>=23.2,<24.0)",
        "Requires-Dist: packaging (>=23.2)",
    ).replace(
        "Requires-Dist: tenacity (>=8.1.0,<9.0.0)",
        "Requires-Dist: tenacity (>=8.1.0)",
    )
    if patched == metadata:
        return f"skip compatible metadata {path.name}"
    metadata_bytes = patched.encode("utf-8")
    digest = base64.urlsafe_b64encode(hashlib.sha256(metadata_bytes).digest()).rstrip(b"=")
    rows = list(csv.reader(io.StringIO(payloads[record_name].decode("utf-8"))))
    for row in rows:
        if row and row[0] == metadata_name:
            row[1] = f"sha256={digest.decode('ascii')}"
            row[2] = str(len(metadata_bytes))
            break
    record_buffer = io.StringIO(newline="")
    csv.writer(record_buffer, lineterminator="\n").writerows(rows)
    payloads[metadata_name] = metadata_bytes
    payloads[record_name] = record_buffer.getvalue().encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with zipfile.ZipFile(tmp_path, "w") as target:
            for info, _payload in entries:
                target.writestr(info, payloads[info.filename])
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return f"patched compatible metadata {path.name}"


def main() -> int:
    packages = _load_lock()
    # Lock packages whose vendored wheel is sourced from PyPI rather than the
    # Pyodide CDN (the CDN has no build for these versions) must be fetched first
    # so the CDN closure loop below skips them as already-present.
    for name, version in PYODIDE_LOCK_PYPI_OVERRIDES.items():
        file_name = f"{name}-{version}-py3-none-any.whl"
        print(_download_pypi_wheel(file_name, VENDOR_DIR))
    package_names = _dependency_closure(_seed_names(packages), packages)
    for name in package_names:
        print(_download(packages[name]))
    for file_name in PYPI_WHEELS:
        print(_download_pypi_wheel(file_name, PYPI_VENDOR_DIR))
        if file_name == LANGCHAIN_CORE_WHEEL:
            print(_patch_langchain_core_metadata(PYPI_VENDOR_DIR / file_name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
