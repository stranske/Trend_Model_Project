"""Minimal typing stub for :mod:`yaml` used in strict mypy runs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import IO, Any, TypeVar

_T = TypeVar("_T")


class YAMLError(Exception):
    ...


class ScannerError(YAMLError):
    ...


class scanner:
    ScannerError = ScannerError


class Loader:
    ...


class SafeLoader(Loader):
    ...


class Dumper:
    ...


class SafeDumper(Dumper):
    ...


def safe_load(stream: str | bytes | bytearray | IO[str] | IO[bytes]) -> Any: ...


def safe_load_all(
    stream: str | bytes | bytearray | IO[str] | IO[bytes],
) -> Iterable[Any]: ...


def safe_dump(
    data: Any,
    stream: IO[str] | None = ...,
    *,
    default_flow_style: bool | None = ...,
    allow_unicode: bool | None = ...,
    sort_keys: bool | None = ...,
) -> str: ...


def dump(
    data: Any,
    stream: IO[str] | None = ...,
    *,
    default_flow_style: bool | None = ...,
    allow_unicode: bool | None = ...,
    sort_keys: bool | None = ...,
) -> str: ...


def load(
    stream: str | bytes | bytearray | IO[str] | IO[bytes],
    Loader: type[Loader] | None = ...,
) -> Any: ...


def dump_all(
    documents: Iterable[Any],
    stream: IO[str] | None = ...,
    Dumper: type[Dumper] | None = ...,
) -> str: ...

