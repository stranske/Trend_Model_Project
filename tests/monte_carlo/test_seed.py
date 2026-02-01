from __future__ import annotations

import numpy as np

from trend_analysis.monte_carlo.seed import SeedManager


def test_path_seed_is_deterministic() -> None:
    manager = SeedManager(42)

    assert manager.get_path_seed(5) == manager.get_path_seed(5)
    assert manager.get_path_seed(5) == SeedManager(42).get_path_seed(5)


def test_strategy_seed_is_deterministic() -> None:
    manager = SeedManager(42)

    assert manager.get_strategy_seed(3, "trend_basic") == manager.get_strategy_seed(
        3, "trend_basic"
    )
    assert manager.get_strategy_seed(3, "trend_basic") == SeedManager(42).get_strategy_seed(
        3, "trend_basic"
    )


def test_get_path_rng_is_reproducible() -> None:
    manager = SeedManager(7)

    rng_a = manager.get_path_rng(1)
    rng_b = manager.get_path_rng(1)

    np.testing.assert_allclose(rng_a.random(8), rng_b.random(8))


def test_get_strategy_rng_is_reproducible() -> None:
    manager = SeedManager(7)

    rng_a = manager.get_strategy_rng(1, "trend_basic")
    rng_b = manager.get_strategy_rng(1, "trend_basic")

    np.testing.assert_allclose(rng_a.normal(size=6), rng_b.normal(size=6))


def test_path_rngs_differ_across_paths() -> None:
    manager = SeedManager(101)

    rng_a = manager.get_path_rng(0)
    rng_b = manager.get_path_rng(1)

    values_a = rng_a.random(6)
    values_b = rng_b.random(6)

    assert not np.array_equal(values_a, values_b)


def test_strategy_rngs_differ_across_strategies() -> None:
    manager = SeedManager(101)

    rng_a = manager.get_strategy_rng(0, "strategy_a")
    rng_b = manager.get_strategy_rng(0, "strategy_b")

    values_a = rng_a.integers(0, 1000, size=10)
    values_b = rng_b.integers(0, 1000, size=10)

    assert not np.array_equal(values_a, values_b)


def test_common_random_numbers_reduce_variance() -> None:
    manager = SeedManager(2026)
    n_paths = 200
    horizon = 50
    noise_scale = 0.1

    common_diffs = []
    independent_diffs = []

    for path_id in range(n_paths):
        path_rng = manager.get_path_rng(path_id)
        base = path_rng.normal(size=horizon)
        noise_a = manager.get_strategy_rng(path_id, "strategy_a").normal(size=horizon)
        noise_b = manager.get_strategy_rng(path_id, "strategy_b").normal(size=horizon)

        common_a = (base + noise_scale * noise_a).mean()
        common_b = (base + noise_scale * noise_b).mean()
        common_diffs.append(common_a - common_b)

        path_rng_a = manager.get_path_rng(path_id * 2 + 1)
        path_rng_b = manager.get_path_rng(path_id * 2 + 2)
        base_a = path_rng_a.normal(size=horizon)
        base_b = path_rng_b.normal(size=horizon)

        independent_a = (base_a + noise_scale * noise_a).mean()
        independent_b = (base_b + noise_scale * noise_b).mean()
        independent_diffs.append(independent_a - independent_b)

    common_variance = np.var(common_diffs)
    independent_variance = np.var(independent_diffs)

    assert common_variance < independent_variance * 0.75
