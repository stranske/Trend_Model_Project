import hashlib

from trend_analysis.util.hash import (
    sha256_bytes,
    sha256_config,
    sha256_file,
    sha256_text,
)


def test_sha256_bytes_matches_hashlib():
    data = b"hello world"
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_sha256_text_matches_hashlib():
    text = "testing"
    assert sha256_text(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_sha256_file_large(tmp_path):
    data = b"A" * (8192 + 100)
    path = tmp_path / "large.bin"
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_sha256_config_order_invariant():
    cfg1 = {"a": 1, "b": 2}
    cfg2 = {"b": 2, "a": 1}
    assert sha256_config(cfg1) == sha256_config(cfg2)
