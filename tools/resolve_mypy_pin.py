from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:  # pragma: no cover - fallback for Python < 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback when tomllib missing
    import tomli as tomllib  # type: ignore[no-redef]


class PinResolutionError(RuntimeError):
    """Raised when the mypy pin cannot be determined safely."""


@dataclass(frozen=True)
class PinResolutionResult:
    pin: str | None
    notices: tuple[tuple[str, str], ...]


def _emit(level: str, message: str, *, file: str | None = None) -> None:
    if file:
        print(f"::{level} file={file}::{message}")
    else:
        print(f"::{level}::{message}")


def _load_pyproject(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(path) from exc
    except OSError as exc:
        raise PinResolutionError(f"Unexpected error reading {path}: {exc}") from exc

    try:
        return tomllib.loads(raw)
    except tomllib.TOMLDecodeError as exc:  # pragma: no cover - pass-through to caller
        raise PinResolutionError(f"Failed to parse {path}: {exc}") from exc


def resolve_pin(pyproject: Path, matrix_version: str | None) -> PinResolutionResult:
    """Resolve the python_version mypy pin following workflow semantics."""

    normalized_matrix = (matrix_version or "").strip()
    notices: list[tuple[str, str]] = []

    if not pyproject.is_file():
        if normalized_matrix:
            notices.append(
                (
                    "notice",
                    "pyproject.toml not found; defaulting mypy python_version to matrix "
                    f"interpreter {normalized_matrix}",
                )
            )
            return PinResolutionResult(normalized_matrix, tuple(notices))
        notices.append(
            (
                "warning",
                "pyproject.toml not found and no matrix interpreter provided; skipping mypy pin resolution",
            )
        )
        return PinResolutionResult(None, tuple(notices))

    data = _load_pyproject(pyproject)
    raw_pin = (
        data.get("tool", {})
        .get("mypy", {})
        .get("python_version")
    )

    if raw_pin is not None:
        pin = str(raw_pin).strip()
        if pin:
            return PinResolutionResult(pin, tuple(notices))

    if normalized_matrix:
        notices.append(
            (
                "notice",
                "No mypy python_version pin configured; defaulting to matrix interpreter "
                f"{normalized_matrix}",
            )
        )
        return PinResolutionResult(normalized_matrix, tuple(notices))

    notices.append(
        (
            "warning",
            "No mypy python_version pin found and matrix version unavailable",
        )
    )
    return PinResolutionResult(None, tuple(notices))


def main(argv: Iterable[str] | None = None) -> int:
    del argv  # unused; retained for CLI compatibility

    pyproject = Path(os.environ.get("PYPROJECT_PATH", "pyproject.toml"))
    matrix_version = os.environ.get("MATRIX_PYTHON_VERSION", "")

    try:
        result = resolve_pin(pyproject, matrix_version)
    except FileNotFoundError:
        if matrix_version:
            _emit(
                "notice",
                "pyproject.toml not found; defaulting mypy python_version to matrix "
                f"interpreter {matrix_version.strip()}",
            )
            pin = matrix_version.strip()
            if pin:
                _write_output(pin)
            return 0
        _emit("warning", "pyproject.toml not found and no matrix interpreter provided; skipping mypy pin resolution")
        return 0
    except PinResolutionError as exc:
        _emit("error", str(exc), file=str(pyproject))
        return 1

    for level, message in result.notices:
        _emit(level, message)

    if result.pin:
        print(f"Resolved mypy python_version pin: {result.pin}")
        _write_output(result.pin)
    return 0


def _write_output(pin: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    path = Path(output_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"python-version={pin}\n")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
