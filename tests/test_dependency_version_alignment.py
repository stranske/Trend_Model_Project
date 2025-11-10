"""Ensure dependency version specifications align across manifests.

This guards against drift between pyproject.toml and requirements.txt so
installation paths cannot resolve conflicting versions.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Dict

_OPERATORS = ("==", ">=", "<=", "~=", "!=", ">", "<", "===")


def _split_spec(raw: str) -> tuple[str, str]:
    entry = raw.strip().strip(",").strip('"')
    for operator in _OPERATORS:
        if operator in entry:
            name, version = entry.split(operator, 1)
            return name.strip(), f"{operator}{version.strip()}"
    return entry.strip(), ""


def _load_requirements_specs(path: Path) -> Dict[str, str]:
    specs: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, spec = _split_spec(stripped)
        specs[name] = spec
    return specs


def _load_pyproject_specs(path: Path) -> Dict[str, str]:
    specs: Dict[str, str] = {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project", {})

    for dependency in project.get("dependencies", []):
        name, spec = _split_spec(dependency)
        specs[name] = spec

    optional = project.get("optional-dependencies", {})
    for group in optional.values():
        for dependency in group:
            name, spec = _split_spec(dependency)
            specs[name] = spec

    return specs


def test_dependency_specifications_are_aligned() -> None:
    requirements_specs = _load_requirements_specs(Path("requirements.txt"))
    pyproject_specs = _load_pyproject_specs(Path("pyproject.toml"))

    mismatches: Dict[str, tuple[str, str]] = {}
    for name in sorted(set(requirements_specs) & set(pyproject_specs)):
        if requirements_specs[name] != pyproject_specs[name]:
            mismatches[name] = (requirements_specs[name], pyproject_specs[name])

    assert not mismatches, (
        "Dependency specifications differ between requirements.txt and pyproject.toml: "
        + ", ".join(
            f"{name} (requirements: '{req}' vs pyproject: '{proj}')"
            for name, (req, proj) in mismatches.items()
        )
    )
