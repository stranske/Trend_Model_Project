#!/usr/bin/env python3
"""Synchronise formatter/test tool version pins across the repository.

This script treats `.github/workflows/autofix-versions.env` as the source of truth
for formatter, lint, and unit-test tooling versions. It can either verify that
`pyproject.toml` and `requirements.txt` match those pins or rewrite the files to
bring them into alignment.

Usage
-----

```
python scripts/sync_tool_versions.py --check   # default, exits 1 on mismatch
python scripts/sync_tool_versions.py --apply   # rewrites files in place
```

The script is intentionally dependency-free so it can run in CI prior to any
third-party installs.
"""
from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

PIN_FILE = Path(".github/workflows/autofix-versions.env")
PYPROJECT_FILE = Path("pyproject.toml")
REQUIREMENTS_FILE = Path("requirements.txt")


@dataclasses.dataclass(frozen=True)
class ToolConfig:
    """Metadata describing how to align a tool's version pins."""

    env_key: str
    package_name: str
    pyproject_pattern: re.Pattern
    pyproject_format: str
    requirements_pattern: re.Pattern
    requirements_format: str


def _compile(pattern: str) -> re.Pattern:
    return re.compile(pattern, flags=re.MULTILINE)


TOOL_CONFIGS: Tuple[ToolConfig, ...] = (
    ToolConfig(
        env_key="BLACK_VERSION",
        package_name="black",
        pyproject_pattern=_compile(r'"black==(?P<version>[^"]+)"'),
        pyproject_format='"black=={version}",',
        requirements_pattern=_compile(r"^black(?:==(?P<version>[^\s#]+))?\s*$"),
        requirements_format="black=={version}",
    ),
    ToolConfig(
        env_key="RUFF_VERSION",
        package_name="ruff",
        pyproject_pattern=_compile(r'"ruff==(?P<version>[^"]+)"'),
        pyproject_format='"ruff=={version}",',
        requirements_pattern=_compile(r"^ruff(?:==(?P<version>[^\s#]+))?\s*$"),
        requirements_format="ruff=={version}",
    ),
    ToolConfig(
        env_key="ISORT_VERSION",
        package_name="isort",
        pyproject_pattern=_compile(r'"isort==(?P<version>[^"]+)"'),
        pyproject_format='"isort=={version}",',
        requirements_pattern=_compile(r"^isort(?:==(?P<version>[^\s#]+))?\s*$"),
        requirements_format="isort=={version}",
    ),
    ToolConfig(
        env_key="DOCFORMATTER_VERSION",
        package_name="docformatter",
        pyproject_pattern=_compile(r'"docformatter==(?P<version>[^"]+)"'),
        pyproject_format='"docformatter=={version}",',
        requirements_pattern=_compile(r"^docformatter(?:==(?P<version>[^\s#]+))?\s*$"),
        requirements_format="docformatter=={version}",
    ),
    ToolConfig(
        env_key="MYPY_VERSION",
        package_name="mypy",
        pyproject_pattern=_compile(r'"mypy(?:==|>=)(?P<version>[^"]+)"'),
        pyproject_format='"mypy=={version}",',
        requirements_pattern=_compile(r"^mypy(?:==(?P<version>[^\s#]+))?\s*$"),
        requirements_format="mypy=={version}",
    ),
    ToolConfig(
        env_key="PYTEST_VERSION",
        package_name="pytest",
        pyproject_pattern=_compile(r'"pytest==(?P<version>[^"]+)"'),
        pyproject_format='"pytest=={version}",',
        requirements_pattern=_compile(r"^pytest(?:==(?P<version>[^\s#]+))?\s*$"),
        requirements_format="pytest=={version}",
    ),
    ToolConfig(
        env_key="PYTEST_COV_VERSION",
        package_name="pytest-cov",
        pyproject_pattern=_compile(r'"pytest-cov==(?P<version>[^"]+)"'),
        pyproject_format='"pytest-cov=={version}",',
        requirements_pattern=_compile(r"^pytest-cov(?:==(?P<version>[^\s#]+))?\s*$"),
        requirements_format="pytest-cov=={version}",
    ),
    ToolConfig(
        env_key="COVERAGE_VERSION",
        package_name="coverage",
        pyproject_pattern=_compile(r'"coverage==(?P<version>[^"]+)"'),
        pyproject_format='"coverage=={version}",',
        requirements_pattern=_compile(r"^coverage(?:==(?P<version>[^\s#]+))?\s*$"),
        requirements_format="coverage=={version}",
    ),
)


class SyncError(RuntimeError):
    """Raised when the repository is misconfigured or a sync fails."""


def parse_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise SyncError(f"Pin file '{path}' does not exist")

    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        values[key.strip()] = raw_value.strip()

    missing = [cfg.env_key for cfg in TOOL_CONFIGS if cfg.env_key not in values]
    if missing:
        raise SyncError(f"Pin file '{path}' is missing keys: {', '.join(missing)}")
    return values


def ensure_pyproject(
    content: str, configs: Iterable[ToolConfig], env: Dict[str, str], apply: bool
) -> Tuple[str, Dict[str, str]]:
    mismatches: Dict[str, str] = {}
    updated_content = content

    for cfg in configs:
        expected = env[cfg.env_key]
        match = cfg.pyproject_pattern.search(updated_content)
        if not match:
            raise SyncError(
                f"pyproject.toml is missing an entry for {cfg.package_name}; expected pattern '{cfg.pyproject_pattern.pattern}'"
            )
        current = match.group("version")
        if current != expected:
            mismatches[cfg.package_name] = (
                f"pyproject has {current}, pin file requires {expected}"
            )
            if apply:
                replacement = cfg.pyproject_pattern.sub(
                    lambda m: cfg.pyproject_format.format(version=expected),
                    updated_content,
                    count=1,
                )
                updated_content = replacement
    return updated_content, mismatches


def ensure_requirements(
    lines: Iterable[str],
    configs: Iterable[ToolConfig],
    env: Dict[str, str],
    apply: bool,
) -> Tuple[str, Dict[str, str]]:
    mismatches: Dict[str, str] = {}
    new_lines = list(lines)

    for cfg in configs:
        expected = env[cfg.env_key]
        matched = False
        for idx, line in enumerate(new_lines):
            match = cfg.requirements_pattern.match(line.strip())
            if not match:
                continue
            matched = True
            current = match.groupdict().get("version")
            if current != expected:
                mismatches[cfg.package_name] = (
                    f"requirements.txt has {current or 'unversioned'}, pin file requires {expected}"
                )
                if apply:
                    new_lines[idx] = cfg.requirements_format.format(version=expected)
            break
        if not matched:
            raise SyncError(
                f"requirements.txt is missing a line for {cfg.package_name}"
            )
    return "\n".join(new_lines) + "\n", mismatches


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Synchronise tool version pins across repository artefacts."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite files to match pinned versions instead of only checking",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Explicitly run in verification-only mode (default)",
    )
    args = parser.parse_args(list(argv))
    apply_changes = args.apply

    if args.check and args.apply:
        parser.error("--apply and --check are mutually exclusive")
    if not args.apply:
        apply_changes = False

    env_values = parse_env_file(PIN_FILE)

    pyproject_content = PYPROJECT_FILE.read_text(encoding="utf-8")
    requirements_content = REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines()

    pyproject_updated, project_mismatches = ensure_pyproject(
        pyproject_content, TOOL_CONFIGS, env_values, apply_changes
    )
    requirements_updated, requirements_mismatches = ensure_requirements(
        requirements_content, TOOL_CONFIGS, env_values, apply_changes
    )

    mismatches = {**project_mismatches, **requirements_mismatches}

    if mismatches and not apply_changes:
        for package, message in mismatches.items():
            print(f"✗ {package}: {message}", file=sys.stderr)
        print("Use --apply to rewrite files with the pinned versions.", file=sys.stderr)
        return 1

    if apply_changes:
        if pyproject_updated != pyproject_content:
            PYPROJECT_FILE.write_text(pyproject_updated, encoding="utf-8")
        if requirements_updated != "\n".join(requirements_content) + "\n":
            REQUIREMENTS_FILE.write_text(requirements_updated, encoding="utf-8")
        print("✓ tool pins synced to pyproject.toml and requirements.txt")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SyncError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
