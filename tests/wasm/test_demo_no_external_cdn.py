"""Offline-runtime guards for the stlite browser demo."""

from __future__ import annotations

import re
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_HTML = REPO_ROOT / "demo" / "wasm" / "index.html"
MANIFEST_PATH = REPO_ROOT / "demo" / "wasm" / "manifest.json"
PYODIDE_VENDOR_DIR = REPO_ROOT / "demo" / "wasm" / "vendor" / "pyodide-0.27.2"
STLITE_VENDOR_DIR = REPO_ROOT / "demo" / "wasm" / "vendor" / "stlite@0.79.4"


def _assert_non_empty(path: Path) -> None:
    assert path.is_file(), f"missing vendored runtime file: {path}"
    assert path.stat().st_size > 0, f"vendored runtime file is empty: {path}"


def _package_name(requirement: str) -> str:
    return re.split(r"[<>=!~;,\[]", requirement, maxsplit=1)[0].strip()


def _lock_key(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pyodide_lock_packages() -> dict[str, dict]:
    lock_path = PYODIDE_VENDOR_DIR / "pyodide-lock.json"
    return json.loads(lock_path.read_text(encoding="utf-8"))["packages"]


def _manifest() -> dict:
    """Return the demo manifest.

    ``demo/wasm/manifest.json`` is a generated, git-ignored build artifact, so it
    is absent on a fresh checkout/CI run. Build it in-process from the
    source-of-truth ``scripts/build_wasm_demo.py`` so this guard never depends on
    a pre-built artifact.
    """
    if MANIFEST_PATH.is_file():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_wasm_demo", REPO_ROOT / "scripts" / "build_wasm_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.build_manifest()


def _dependency_closure(seed_names: list[str], packages: dict[str, dict]) -> set[str]:
    closure: set[str] = set()
    stack = [_lock_key(name) for name in seed_names]
    while stack:
        name = stack.pop()
        if name in closure:
            continue
        assert name in packages, f"requirement not found in pyodide lock: {name}"
        closure.add(name)
        stack.extend(_lock_key(dep) for dep in packages[name].get("depends", []))
    return closure


def _runtime_refs(html: str) -> list[str]:
    refs: list[str] = []
    refs.extend(
        match.group("url")
        for match in re.finditer(
            r"<script\b[^>]*\bsrc=[\"'](?P<url>https?://[^\"']+)[\"']",
            html,
            flags=re.IGNORECASE,
        )
    )
    refs.extend(
        match.group("url")
        for match in re.finditer(
            r"<link\b[^>]*\bhref=[\"'](?P<url>https?://[^\"']+)[\"']",
            html,
            flags=re.IGNORECASE,
        )
    )
    refs.extend(
        match.group("url")
        for match in re.finditer(
            r"\bimport\b[^;]*?\bfrom\s+[\"'](?P<url>https?://[^\"']+)[\"']",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    return refs


def test_demo_html_has_no_external_runtime_ref() -> None:
    html = DEMO_HTML.read_text(encoding="utf-8")
    assert _runtime_refs(html) == []
    assert "cdn.jsdelivr.net" not in html
    assert "./vendor/stlite@0.79.4/stlite.js" in html
    assert "./vendor/pyodide-0.27.2/pyodide.mjs" in html


def test_pyodide_runtime_is_vendored() -> None:
    for name in (
        "pyodide.mjs",
        "pyodide.asm.wasm",
        "python_stdlib.zip",
        "pyodide-lock.json",
    ):
        _assert_non_empty(PYODIDE_VENDOR_DIR / name)


def test_stlite_runtime_is_vendored() -> None:
    for name in ("stlite.js", "style.css"):
        _assert_non_empty(STLITE_VENDOR_DIR / name)

    streamlit_wheels = list((STLITE_VENDOR_DIR / "wheels").glob("streamlit-*.whl"))
    stlite_lib_wheels = list((STLITE_VENDOR_DIR / "wheels").glob("stlite_lib-*.whl"))
    assert streamlit_wheels, "missing vendored streamlit wheel"
    assert stlite_lib_wheels, "missing vendored stlite_lib wheel"
    for path in streamlit_wheels + stlite_lib_wheels:
        _assert_non_empty(path)


def test_presentation_safe_pyodide_dependency_closure_is_vendored() -> None:
    manifest = _manifest()
    requirements = manifest["requirements"]["presentation_safe"]
    seed_names = [_package_name(requirement) for requirement in requirements]
    packages = _pyodide_lock_packages()

    missing: list[str] = []
    for name in sorted(_dependency_closure(seed_names, packages)):
        file_name = packages[name]["file_name"]
        if not (PYODIDE_VENDOR_DIR / file_name).is_file():
            missing.append(f"{name}: {file_name}")

    assert missing == []
