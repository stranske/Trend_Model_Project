#!/usr/bin/env python3
"""Regenerate the committed Pyodide runtime artifacts for the stlite demo.

The wheels are COMMITTED so the demo boots offline with zero network access;
this script regenerates or updates them from the Pyodide CDN when maintainers
intentionally refresh the vendored runtime. Plotly is not in the Pyodide lock
(it is pure-PyPI for this runtime) and is intentionally excluded.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import urllib.request
from pathlib import Path

from build_wasm_demo import REPO_ROOT, REQUIREMENTS

PYODIDE_VERSION = "0.27.2"
CDN_BASE_URL = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full"
VENDOR_DIR = REPO_ROOT / "demo" / "wasm" / "vendor" / f"pyodide-{PYODIDE_VERSION}"
LOCK_PATH = VENDOR_DIR / "pyodide-lock.json"


def _package_name(requirement: str) -> str:
    return re.split(r"[<>=!~;,\[]", requirement, maxsplit=1)[0].strip()


def _lock_key(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _load_lock() -> dict[str, dict]:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))["packages"]


def _vendored_package_names(packages: dict[str, dict]) -> set[str]:
    vendored_files = {path.name for path in VENDOR_DIR.iterdir() if path.is_file()}
    return {
        name
        for name, package in packages.items()
        if package.get("file_name") in vendored_files
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
        for name in (
            manifest_requirement_names | pyodide_runtime_names | already_vendored_names
        )
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


def _download(package: dict[str, str]) -> str:
    file_name = package["file_name"]
    target = VENDOR_DIR / file_name
    if target.is_file():
        return f"skip existing {file_name}"

    url = f"{CDN_BASE_URL}/{file_name}"
    with tempfile.NamedTemporaryFile(dir=VENDOR_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(url) as response:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)

    expected_sha = package.get("sha256")
    if expected_sha and _sha256(tmp_path) != expected_sha:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {file_name}")

    tmp_path.replace(target)
    return f"downloaded {file_name}"


def main() -> int:
    packages = _load_lock()
    package_names = _dependency_closure(_seed_names(packages), packages)
    for name in package_names:
        print(_download(packages[name]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
