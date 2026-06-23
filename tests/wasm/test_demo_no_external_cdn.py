"""Offline-runtime guards for the stlite browser demo."""

from __future__ import annotations

import json
import re
import zipfile
from email.parser import Parser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "demo" / "wasm"
DEMO_HTML = DEMO_DIR / "index.html"
MANIFEST_PATH = DEMO_DIR / "manifest.json"
PYODIDE_VENDOR_DIR = DEMO_DIR / "vendor" / "pyodide-0.27.2"
STLITE_VENDOR_DIR = DEMO_DIR / "vendor" / "stlite@0.79.4"
PYPI_VENDOR_DIR = DEMO_DIR / "vendor" / "pypi"

#: Lowest narwhals the plotly 6.x ``plotly.express`` path needs (it calls
#: ``narwhals.from_native(..., pass_through=...)``, absent in the lock's 1.10.0).
MIN_NARWHALS = (1, 15, 1)


def _assert_non_empty(path: Path) -> None:
    assert path.is_file(), f"missing vendored runtime file: {path}"
    assert path.stat().st_size > 0, f"vendored runtime file is empty: {path}"


def _is_wheel_requirement(requirement: str) -> bool:
    """A requirement that is a vendored wheel path rather than a PyPI name."""
    return requirement.endswith(".whl")


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


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


def test_stlite_wheel_metadata_dependencies_are_vendored() -> None:
    wheel_dir = STLITE_VENDOR_DIR / "wheels"
    wheel_files = {path.name for path in wheel_dir.glob("*.whl")}
    streamlit_wheels = list(wheel_dir.glob("streamlit-*.whl"))
    assert streamlit_wheels, "missing vendored streamlit wheel"

    with zipfile.ZipFile(streamlit_wheels[0]) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))

    pure_python_deps = {
        _lock_key(_package_name(requirement))
        for requirement in metadata.get_all("Requires-Dist", [])
        if _lock_key(_package_name(requirement)) in {"blinker", "tenacity"}
    }
    missing = [
        name
        for name in sorted(pure_python_deps)
        if not any(file_name.startswith(f"{name}-") for file_name in wheel_files)
    ]
    assert missing == []


def test_presentation_safe_pyodide_dependency_closure_is_vendored() -> None:
    manifest = _manifest()
    packages = _pyodide_lock_packages()
    # Wheel-path requirements (plotly, under vendor/pypi/) are not in the Pyodide
    # lock; they are excluded here and asserted separately in
    # test_plotly_wheel_is_vendored.
    seed_names = [
        _package_name(requirement)
        for requirement in manifest["requirements"]["presentation_safe"]
        if not _is_wheel_requirement(requirement)
    ]
    seed_names.extend(["micropip", "packaging"])
    missing_from_lock = sorted(
        {name for name in seed_names if _lock_key(name) not in packages}
    )
    assert missing_from_lock == [], (
        f"requirements missing from pyodide lock: {missing_from_lock}"
    )
    seed_names = [_lock_key(name) for name in seed_names]

    missing: list[str] = []
    for name in sorted(_dependency_closure(seed_names, packages)):
        file_name = packages[name]["file_name"]
        if not (PYODIDE_VENDOR_DIR / file_name).is_file():
            missing.append(f"{name}: {file_name}")

    assert missing == []


def test_plotly_wheel_is_vendored() -> None:
    """plotly is pure-PyPI (not in the lock); it is committed under vendor/pypi/
    and listed in every profile's requirements as a repo-relative wheel path."""
    manifest = _manifest()
    for profile, requirements in manifest["requirements"].items():
        wheels = [r for r in requirements if _is_wheel_requirement(r)]
        plotly_wheels = [w for w in wheels if "plotly-" in w]
        assert plotly_wheels, f"{profile}: no plotly wheel requirement ({wheels})"
        for wheel in wheels:
            # repo-relative path under demo/wasm/, NOT a bare/absolute/CDN URL —
            # index.html resolves these to absolute same-origin URLs at runtime.
            assert wheel.startswith("vendor/pypi/"), wheel
            assert not wheel.startswith(("http://", "https://", "./", "/")), wheel
            _assert_non_empty(DEMO_DIR / wheel)


def test_index_html_resolves_wheel_requirements_to_absolute_urls() -> None:
    """The stlite worker's micropip rejects a bare-relative wheel URL (treats it
    as file://), so index.html must rewrite every ``*.whl`` requirement to an
    absolute same-origin URL via ``new URL(req, window.location.href)``."""
    html = DEMO_HTML.read_text(encoding="utf-8")
    assert re.search(r"""endsWith\(\s*["']\.whl["']\s*\)""", html), html
    assert re.search(
        r"new URL\(\s*\w+\s*,\s*window\.location\.href\s*\)", html
    ), html


def test_lock_narwhals_is_bumped_for_plotly_express() -> None:
    """plotly 6.x's plotly.express calls narwhals.from_native(pass_through=...),
    which the Pyodide 0.27.2 lock's narwhals 1.10.0 lacks. The lock is bumped in
    place to >=1.15.1 (a single narwhals also serves the lock's altair)."""
    packages = _pyodide_lock_packages()
    narwhals = packages["narwhals"]
    assert _version_tuple(narwhals["version"]) >= MIN_NARWHALS, narwhals["version"]
    assert narwhals["version"] in narwhals["file_name"], narwhals
    _assert_non_empty(PYODIDE_VENDOR_DIR / narwhals["file_name"])


def test_monte_carlo_scenario_configs_are_bundled() -> None:
    """The Monte Carlo page loads its scenario registry from
    ``config/scenarios/monte_carlo/index.yml`` and each scenario's
    ``base_config`` (``config/defaults.yml``). These non-Python config files must
    be bundled into the browser FS for the page to run a scenario offline."""
    manifest = _manifest()
    files = set(manifest["files"])
    required = {
        "config/defaults.yml",
        "config/scenarios/monte_carlo/index.yml",
    }
    missing = sorted(required - files)
    assert missing == [], f"unbundled Monte Carlo config: {missing}"
    # At least one runnable scenario yml beyond the index is bundled.
    scenario_files = [
        f
        for f in files
        if f.startswith("config/scenarios/monte_carlo/")
        and f.endswith(".yml")
        and not f.endswith("index.yml")
    ]
    assert scenario_files, "no Monte Carlo scenario ymls bundled"
    # The manifest lists repo-relative paths; the deploy publishes these source
    # files under the app base. Verify the source files the manifest points to
    # actually exist and are non-empty.
    for rel in sorted(required | set(scenario_files)):
        _assert_non_empty(REPO_ROOT / rel)
