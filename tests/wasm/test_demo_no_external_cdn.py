"""Offline-runtime guards for the stlite browser demo."""

from __future__ import annotations

import json
import re
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


def test_presentation_safe_pyodide_dependency_closure_is_vendored() -> None:
    manifest = _manifest()
    packages = _pyodide_lock_packages()
    # Wheel-path requirements (plotly, under vendor/pypi/) are not in the Pyodide
    # lock, so they are dropped by the ``_lock_key(name) in packages`` filter
    # below and asserted separately in test_plotly_wheel_is_vendored.
    seed_names = [
        _package_name(requirement)
        for requirements in manifest["requirements"].values()
        for requirement in requirements
    ]
    seed_names.extend(["micropip", "packaging"])
    seed_names = [name for name in seed_names if _lock_key(name) in packages]

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


MC_REGISTRY_REL = "config/scenarios/monte_carlo/index.yml"


def _resolve_repo_relative(ref: str, *, base_file_rel: str) -> str | None:
    """Mirror the registry's path resolution and return a repo-relative posix path.

    The scenario registry resolves a referenced ``ref`` against the referencing
    file's directory, its parent, and the repo root (see
    ``trend_analysis.monte_carlo.registry._resolve_path`` /
    ``_resolve_base_config`` / ``_resolve_strategy_pack``). Replicate that here so
    the guard tracks the same files the runtime will read. Returns ``None`` if no
    candidate exists on disk.
    """
    base_dir = (REPO_ROOT / base_file_rel).parent
    candidates = [base_dir / ref, base_dir.parent / ref, REPO_ROOT / ref]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved.relative_to(REPO_ROOT).as_posix()
    return None


def test_monte_carlo_scenario_configs_are_bundled() -> None:
    """The Monte Carlo page lists/loads scenarios from
    ``config/scenarios/monte_carlo/index.yml`` via ``proj_path`` (the FS root the
    stlite manifest mounts files at). If the registry, the scenarios it
    references, their ``base_config``, and their curated strategy packs are not in
    the manifest, the page raises "Scenario registry '...' does not exist" (the
    pre-fix offline failure for issue #5643). Assert the whole reachable config
    closure is bundled.
    """
    import yaml

    manifest = _manifest()
    files = set(manifest["files"])

    # Anchors that must always be present.
    for anchor in (
        MC_REGISTRY_REL,
        "config/scenarios/example_scenario.yml",
        "config/defaults.yml",
        "config/scenarios/monte_carlo/strategies/hf_equity_curated.yml",
    ):
        assert anchor in files, f"{anchor} missing from demo manifest files"

    # Every scenario the registry references -- plus its base_config and curated
    # strategy pack -- must be bundled too, so newly registered scenarios that
    # the build glob would miss fail this guard.
    registry = yaml.safe_load((REPO_ROOT / MC_REGISTRY_REL).read_text(encoding="utf-8"))
    entries = registry["scenarios"]
    assert isinstance(entries, list) and entries

    for entry in entries:
        scenario_rel = _resolve_repo_relative(entry["path"], base_file_rel=MC_REGISTRY_REL)
        assert scenario_rel is not None, f"registry path {entry['path']!r} not found on disk"
        assert scenario_rel in files, f"scenario {scenario_rel} missing from demo manifest files"

        scenario = yaml.safe_load((REPO_ROOT / scenario_rel).read_text(encoding="utf-8"))

        base_config = scenario.get("base_config")
        if isinstance(base_config, str):
            base_rel = _resolve_repo_relative(base_config, base_file_rel=scenario_rel)
            assert base_rel is not None, f"{scenario_rel}: base_config {base_config!r} not found"
            assert base_rel in files, f"base_config {base_rel} missing from demo manifest files"

        strategy_set = scenario.get("strategy_set")
        if isinstance(strategy_set, dict):
            pack = strategy_set.get("curated_pack")
            if isinstance(pack, str):
                pack_rel = _resolve_repo_relative(pack, base_file_rel=scenario_rel)
                assert pack_rel is not None, f"{scenario_rel}: curated_pack {pack!r} not found"
                assert pack_rel in files, f"curated_pack {pack_rel} missing from manifest files"
