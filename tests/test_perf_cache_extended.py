import numpy as np
import pandas as pd
import pytest

from trend_analysis.perf.cache import (
    CovCache,
    CovPayload,
    _ensure_aggregates,
    compute_cov_payload,
    incremental_cov_update,
)


def make_df(rows: int = 5, cols: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = rng.normal(0, 0.01, size=(rows, cols))
    columns = [f"F{i}" for i in range(cols)]
    return pd.DataFrame(data, columns=columns)


def test_covcache_tracks_hits_misses_and_eviction() -> None:
    df = make_df()
    cache = CovCache(capacity=1)
    key1 = cache.make_key("2020-01", "2020-03", df.columns)

    # Initial lookup is a miss and increments counter.
    assert cache.get(key1) is None
    assert cache.misses == 1

    payload1 = compute_cov_payload(df)
    cache.put(key1, payload1)

    # Retrieve -> hit path moves entry to end and increments hit count.
    assert cache.get(key1) is payload1
    assert cache.hits == 1

    # Re-put same key triggers move_to_end branch without changing size.
    cache.put(key1, payload1)
    assert len(cache) == 1

    # Adding a different key pushes us past capacity and evicts the oldest.
    key2 = cache.make_key("2020-04", "2020-06", df.columns)
    payload2 = compute_cov_payload(df * 2)
    cache.put(key2, payload2)
    assert len(cache) == 1

    # Evicted entry causes another miss when requested again.
    assert cache.get(key1) is None
    assert cache.misses == 2


def test_covcache_get_or_compute_without_lru_moves() -> None:
    df = make_df()
    cache = CovCache(lru=False)
    key = cache.make_key("2021-01", "2021-06", df.columns)

    payload1 = cache.get_or_compute(key, lambda: compute_cov_payload(df))
    payload2 = cache.get_or_compute(key, lambda: compute_cov_payload(df * 3))

    # Without LRU the cached payload is returned but hits counter remains zero.
    assert payload1 is payload2
    assert cache.hits == 0
    assert cache.misses == 1  # only the initial miss is recorded


def test_compute_cov_payload_single_row_returns_zero_covariance() -> None:
    df = pd.DataFrame([[0.1, -0.2]], columns=["A", "B"])
    payload = compute_cov_payload(df)

    np.testing.assert_allclose(payload.cov, np.zeros((2, 2)))
    np.testing.assert_allclose(payload.std, np.zeros(2))
    assert payload.n == 1


def test_compute_cov_payload_materialises_aggregates() -> None:
    df = make_df(rows=6, cols=4)
    payload = compute_cov_payload(df, materialise_aggregates=True)

    assert payload.s1 is not None
    assert payload.s2 is not None
    np.testing.assert_allclose(payload.s1, df.to_numpy().sum(axis=0))
    np.testing.assert_allclose(payload.s2, df.to_numpy().T @ df.to_numpy())


def test_ensure_aggregates_populates_missing_values() -> None:
    df = make_df(rows=5, cols=2)
    payload = compute_cov_payload(df)
    assert payload.s1 is None and payload.s2 is None

    _ensure_aggregates(payload)

    assert payload.s1 is not None
    assert payload.s2 is not None
    np.testing.assert_allclose(payload.s1, df.to_numpy().sum(axis=0))


def test_ensure_aggregates_rejects_invalid_sample_size() -> None:
    payload = CovPayload(
        cov=np.zeros((1, 1)),
        mean=np.zeros(1),
        std=np.zeros(1),
        n=0,
        assets=("A",),
    )

    with pytest.raises(ValueError):
        _ensure_aggregates(payload)


def test_incremental_cov_update_validates_shapes_and_sample_size() -> None:
    df = make_df(rows=4, cols=3)
    payload_short = compute_cov_payload(df.iloc[:1])

    with pytest.raises(ValueError):
        incremental_cov_update(
            payload_short, df.iloc[0].to_numpy(), df.iloc[1].to_numpy()
        )

    payload_window = compute_cov_payload(df.iloc[:3])
    bad_new_row = np.append(df.iloc[1].to_numpy(), 0.0)
    with pytest.raises(ValueError):
        incremental_cov_update(payload_window, df.iloc[0].to_numpy(), bad_new_row)

    with pytest.raises(ValueError):
        incremental_cov_update(
            payload_window, df.iloc[0].to_numpy()[:-1], df.iloc[1].to_numpy()
        )


def test_incremental_cov_update_matches_full_recompute() -> None:
    df = make_df(rows=5, cols=2)
    prev_window = compute_cov_payload(df.iloc[:4])
    assert prev_window.s1 is None and prev_window.s2 is None

    updated = incremental_cov_update(
        prev_window,
        df.iloc[0].to_numpy(),
        df.iloc[4].to_numpy(),
    )

    recomputed = compute_cov_payload(df.iloc[1:], materialise_aggregates=True)

    np.testing.assert_allclose(updated.cov, recomputed.cov)
    np.testing.assert_allclose(updated.mean, recomputed.mean)
    np.testing.assert_allclose(updated.std, recomputed.std)
    assert updated.n == recomputed.n
    assert updated.assets == recomputed.assets
    assert updated.s1 is not None and updated.s2 is not None


def test_covcache_emits_timing_logs(monkeypatch) -> None:
    cache = CovCache()
    key = cache.make_key("2020-01", "2020-06", ["A", "B"])

    logged: list[tuple[str, dict[str, object]]] = []

    def fake_log(stage: str, **fields: object) -> None:  # pragma: no cover - simple
        logged.append((stage, fields))

    monkeypatch.setattr("trend_analysis.perf.cache.log_timing", fake_log)

    payload = CovPayload(
        cov=np.eye(2),
        mean=np.zeros(2),
        std=np.zeros(2),
        n=2,
        assets=("A", "B"),
    )

    cache.get_or_compute(key, lambda: payload)
    cache.get_or_compute(key, lambda: payload)

    assert [entry[1]["status"] for entry in logged] == ["miss", "hit"]
