"""Supplementary coverage for :mod:`trend_analysis.perf.cache`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.perf import cache


def test_covpayload_as_dict_and_cache_clear() -> None:
    payload = cache.CovPayload(
        cov=np.eye(2),
        mean=np.array([0.1, 0.2]),
        std=np.array([0.3, 0.4]),
        n=5,
        assets=("A", "B"),
    )
    summary = payload.as_dict()
    assert summary["cov"].shape == (2, 2)
    assert summary["assets"] == ("A", "B")

    store = cache.CovCache()
    key = store.make_key("2020-01", "2020-03", ["A", "B"])
    store.put(key, payload)
    assert len(store.get(key).assets) == 2  # type: ignore[union-attr]
    store.clear()
    assert store.get(key) is None
    # ``get`` registers a miss because the cache entry was cleared.
    assert store.hits == 0
    assert store.misses == 1
    assert store.incremental_updates == 0


def test_compute_cov_payload_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match="DataFrame is empty"):
        cache.compute_cov_payload(pd.DataFrame())


def test_incremental_cov_update_validates_row_length() -> None:
    df = pd.DataFrame({"A": [0.1, 0.2, 0.3], "B": [0.0, 0.1, 0.2]})
    payload = cache.compute_cov_payload(df, materialise_aggregates=True)

    with pytest.raises(ValueError, match="Row length does not match"):
        cache.incremental_cov_update(payload, np.array([0.1]), np.array([0.2]))
