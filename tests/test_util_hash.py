import hashlib
from pathlib import Path

from trend_analysis.util import hash as hash_utils


def test_sha256_bytes_matches_hashlib():
    data = b"trend-analysis"
    assert hash_utils.sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_sha256_text_roundtrip():
    text = "Coverage makes refactors safer"
    assert hash_utils.sha256_text(text) == hash_utils.sha256_bytes(text.encode("utf-8"))


def test_sha256_file_reads_in_chunks(tmp_path: Path):
    content = b"streaming data" * 1024
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    assert hash_utils.sha256_file(file_path) == expected


def test_sha256_config_sorts_keys():
    first = {"alpha": 1, "beta": [1, 2, 3], "nested": {"gamma": True}}
    second = {"nested": {"gamma": True}, "beta": [1, 2, 3], "alpha": 1}

    digest_a = hash_utils.sha256_config(first)
    digest_b = hash_utils.sha256_config(second)
    assert digest_a == digest_b

    # Changing a value should change the hash
    third = dict(first)
    third["alpha"] = 2
    assert hash_utils.sha256_config(third) != digest_a
