from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, TextIO

class YAMLError(Exception): ...

Loader: type[Any]
FullLoader: type[Any]
SafeLoader: type[Any]
Dumper: type[Any]
SafeDumper: type[Any]
CDumper: type[Any]
CSafeDumper: type[Any]

_DefStream = str | bytes | bytearray | TextIO

def safe_load(stream: _DefStream | Iterable[str]) -> Any: ...
def safe_load_all(stream: _DefStream | Iterable[str]) -> Iterable[Any]: ...
def load(stream: _DefStream | Iterable[str], Loader: type[Any] | None = ...) -> Any: ...
def dump(
    data: Any,
    stream: TextIO | None = ...,
    *,
    Dumper: type[Any] | None = ...,
    default_flow_style: bool | None = ...,
    sort_keys: bool | None = ...,
    allow_unicode: bool | None = ...,
) -> str: ...
def safe_dump(
    data: Any,
    stream: TextIO | None = ...,
    *,
    default_flow_style: bool | None = ...,
    sort_keys: bool = ...,
    allow_unicode: bool = ...,
) -> str: ...
def dump_all(
    documents: Iterable[Any],
    stream: TextIO | None = ...,
    *,
    Dumper: type[Any] | None = ...,
    default_flow_style: bool | None = ...,
    sort_keys: bool | None = ...,
    allow_unicode: bool | None = ...,
) -> str: ...
def safe_dump_all(
    documents: Iterable[Any],
    stream: TextIO | None = ...,
    *,
    default_flow_style: bool | None = ...,
    sort_keys: bool | None = ...,
    allow_unicode: bool | None = ...,
) -> str: ...
def add_constructor(
    tag: str, constructor: Any, Loader: type[Any] | None = ...
) -> None: ...
def add_representer(
    tag: type[Any], representer: Any, Dumper: type[Any] | None = ...
) -> None: ...
def add_multi_constructor(
    tag_prefix: str, constructor: Any, Loader: type[Any] | None = ...
) -> None: ...
def add_multi_representer(
    tag: type[Any], representer: Any, Dumper: type[Any] | None = ...
) -> None: ...
def full_load(stream: _DefStream | Iterable[str]) -> Any: ...
def full_load_all(stream: _DefStream | Iterable[str]) -> Iterable[Any]: ...
def safe_stream(stream: TextIO | None = ...) -> Mapping[str, Any]: ...
def scan(stream: _DefStream | Iterable[str]) -> Sequence[Any]: ...
def compose(stream: _DefStream | Iterable[str]) -> Any: ...
