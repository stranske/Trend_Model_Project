"""Regression tests for :mod:`trend_analysis.util.hash`."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from trend_analysis.util import hash as hash_util


class DummyModel:
    """Stand-in object exposing ``model_dump`` for normalisation tests."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self._payload


def test_sha256_bytes_matches_hashlib() -> None:
    data = b"trend-model"
    expected = hashlib.sha256(data).hexdigest()
    assert hash_util.sha256_bytes(data) == expected


def test_sha256_text_round_trips_utf8() -> None:
    text = "Δtrend"
    assert hash_util.sha256_text(text) == hash_util.sha256_bytes(text.encode("utf-8"))


def test_sha256_file_reads_stream(tmp_path: Path) -> None:
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(b"abc" * 1000)

    expected = hash_util.sha256_bytes(file_path.read_bytes())
    assert hash_util.sha256_file(file_path) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Path("/tmp/demo"), "/tmp/demo"),
        ({"alpha": Path("/tmp/a"), "beta": 2}, {"alpha": "/tmp/a", "beta": 2}),
        ([Path("/tmp/b"), 3], ["/tmp/b", 3]),
    ],
)
def test_normalise_for_json_handles_paths_and_collections(value: Any, expected: Any) -> None:
    assert hash_util.normalise_for_json(value) == expected


def test_normalise_for_json_uses_model_dump() -> None:
    model = DummyModel({"threshold": 0.5, "paths": [Path("/data")]})
    normalised = hash_util.normalise_for_json(model)
    assert normalised == {"threshold": 0.5, "paths": ["/data"]}


def test_sha256_config_sorts_keys() -> None:
    cfg_a = {"beta": 2, "alpha": 1}
    cfg_b = {"alpha": 1, "beta": 2}
    assert hash_util.sha256_config(cfg_a) == hash_util.sha256_config(cfg_b)


def test_sha256_config_normalises_nested_values(tmp_path: Path) -> None:
    cfg = {
        "output": tmp_path / "export.csv",
        "values": [1, Path("/var/tmp")],
        "model": DummyModel({"enabled": True, "path": Path("/foo")}),
    }
    digest = hash_util.sha256_config(cfg)

    reloaded = {
        "output": tmp_path / "export.csv",
        "values": [1, Path("/var/tmp")],
        "model": DummyModel({"enabled": True, "path": Path("/foo")}),
    }
    assert digest == hash_util.sha256_config(reloaded)
