from __future__ import annotations

import logging
import random

import pytest

from trend_analysis.monte_carlo.strategy.sampler import (
    CategoricalDistribution,
    DiscreteDistribution,
    UniformDistribution,
    parse_distribution,
    parse_sampling_config,
    sample_strategy_variants,
)


def test_parse_categorical_distribution() -> None:
    dist = parse_distribution(
        {"dist": "categorical", "values": [8, 12, None]}, path="sampling.rank.n"
    )

    assert isinstance(dist, CategoricalDistribution)
    assert dist.values == (8, 12, None)


def test_parse_discrete_distribution_values() -> None:
    dist = parse_distribution(
        {"dist": "discrete", "values": ["equal", "inverse"]}, path="sampling.weighting"
    )

    assert isinstance(dist, DiscreteDistribution)
    assert dist.values == ("equal", "inverse")


def test_parse_discrete_distribution_range() -> None:
    dist = parse_distribution(
        {"dist": "discrete", "low": 1, "high": 3, "step": 1}, path="sampling.rank.n"
    )

    assert isinstance(dist, DiscreteDistribution)
    assert dist.values == (1, 2, 3)


def test_parse_discrete_distribution_range_float_step() -> None:
    dist = parse_distribution(
        {"dist": "discrete", "low": 0.5, "high": 1.5, "step": 0.5},
        path="sampling.weighting",
    )

    assert isinstance(dist, DiscreteDistribution)
    assert dist.values == (0.5, 1.0, 1.5)


def test_parse_discrete_distribution_invalid_step() -> None:
    with pytest.raises(ValueError, match="step must be > 0"):
        parse_distribution(
            {"dist": "discrete", "low": 1, "high": 2, "step": 0},
            path="sampling.weighting",
        )


def test_parse_uniform_distribution() -> None:
    dist = parse_distribution({"dist": "uniform", "low": 0.1, "high": 0.2}, path="sampling.tc")

    assert isinstance(dist, UniformDistribution)
    assert dist.low == 0.1
    assert dist.high == 0.2


def test_discrete_distribution_sample_reproducible() -> None:
    dist = DiscreteDistribution(values=(10, 20, 30))
    rng = random.Random(4)

    assert dist.sample(rng) == 10


def test_categorical_distribution_sample_reproducible() -> None:
    dist = CategoricalDistribution(values=("alpha", "beta", "gamma"))
    rng = random.Random(11)

    assert dist.sample(rng) == "beta"


def test_uniform_distribution_sample_fixed_value() -> None:
    dist = UniformDistribution(low=0.2, high=0.2)

    value = dist.sample(random.Random(123))

    assert value == 0.2


def test_uniform_distribution_sample_within_bounds() -> None:
    dist = UniformDistribution(low=0.1, high=0.2)

    value = dist.sample(random.Random(9))

    assert 0.1 <= value <= 0.2


def test_parse_sampling_config_rejects_empty_segments() -> None:
    with pytest.raises(ValueError, match="empty path segments"):
        parse_sampling_config({"portfolio.": {"dist": "categorical", "values": [1]}})


def test_parse_sampling_config_rejects_empty_mapping() -> None:
    with pytest.raises(ValueError, match="must define at least one distribution"):
        parse_sampling_config({})


def test_sample_strategy_variants_reproducible_seed() -> None:
    sampling = {
        "portfolio.rank.n": {"dist": "categorical", "values": [8, 12]},
        "portfolio.max_turnover": {"dist": "uniform", "low": 0.05, "high": 0.1},
        "portfolio.weighting_scheme": {"dist": "discrete", "values": ["equal", "inverse"]},
    }

    first = sample_strategy_variants(sampling, 3, seed=42)
    second = sample_strategy_variants(sampling, 3, seed=42)

    assert [variant.overrides for variant in first] == [variant.overrides for variant in second]
    assert [variant.name for variant in first] == [variant.name for variant in second]


def test_sample_strategy_variants_constraints_rejects() -> None:
    sampling = {
        "portfolio.rank.n": {"dist": "categorical", "values": [8, 12]},
    }

    def constraint(overrides: dict[str, object]) -> tuple[bool, str]:
        value = overrides["portfolio"]["rank"]["n"]
        return value == 8, "rank.n must be 8"

    variants = sample_strategy_variants(sampling, 1, seed=7, constraints=[constraint])

    assert variants[0].overrides["portfolio"]["rank"]["n"] == 8


def test_sample_strategy_variants_constraints_bool() -> None:
    sampling = {
        "portfolio.rank.n": {"dist": "categorical", "values": [8]},
    }

    def constraint(overrides: dict[str, object]) -> bool:
        return overrides["portfolio"]["rank"]["n"] == 8

    variants = sample_strategy_variants(sampling, 1, seed=5, constraints=[constraint])

    assert variants[0].overrides["portfolio"]["rank"]["n"] == 8


def test_sample_strategy_variants_rejection_logging(caplog: pytest.LogCaptureFixture) -> None:
    sampling = {
        "portfolio.rank.n": {"dist": "categorical", "values": [8, 12]},
    }

    def constraint(overrides: dict[str, object]) -> tuple[bool, str]:
        value = overrides["portfolio"]["rank"]["n"]
        return value == 8, "rank.n must be 8"

    caplog.set_level(logging.INFO, logger="trend_analysis.monte_carlo.strategy.sampler")
    sample_strategy_variants(sampling, 1, seed=7, constraints=[constraint])

    assert "Rejected sampled config" in caplog.text


def test_sample_strategy_variants_max_rejection_attempts(caplog: pytest.LogCaptureFixture) -> None:
    sampling = {"portfolio.rank.n": {"dist": "categorical", "values": [8]}}

    def constraint(_: dict[str, object]) -> tuple[bool, str]:
        return False, "always reject"

    caplog.set_level(logging.INFO, logger="trend_analysis.monte_carlo.strategy.sampler")
    with pytest.raises(RuntimeError, match="max_rejection_attempts"):
        sample_strategy_variants(
            sampling, 1, seed=1, constraints=[constraint], max_rejection_attempts=2
        )

    assert "Rejected sampled config" in caplog.text


def test_sample_strategy_variants_duplicate_rejection_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sampling = {"portfolio.rank.n": {"dist": "uniform", "low": 1.0, "high": 1.0}}

    caplog.set_level(logging.INFO, logger="trend_analysis.monte_carlo.strategy.sampler")
    with pytest.raises(RuntimeError, match="max_rejection_attempts"):
        sample_strategy_variants(sampling, 2, seed=3, max_rejection_attempts=2)

    assert "Rejected duplicate sampled config" in caplog.text


def test_sample_strategy_variants_duplicate_rejection_allows_recovery() -> None:
    sampling = {"portfolio.rank.n": {"dist": "categorical", "values": [1, 2]}}

    variants = sample_strategy_variants(sampling, 2, seed=1, max_rejection_attempts=5)

    values = {variant.overrides["portfolio"]["rank"]["n"] for variant in variants}
    assert values == {1, 2}


def test_sample_strategy_variants_unique_names() -> None:
    sampling = {"portfolio.rank.n": {"dist": "categorical", "values": [1, 2, 3]}}

    variants = sample_strategy_variants(sampling, 3, seed=1, name_prefix="variant")

    names = [variant.name for variant in variants]
    assert names == ["variant_001", "variant_002", "variant_003"]


def test_sample_strategy_variants_respects_existing_names() -> None:
    sampling = {"portfolio.rank.n": {"dist": "categorical", "values": [1, 2, 3]}}

    variants = sample_strategy_variants(
        sampling, 2, seed=1, name_prefix="variant", existing_names=["variant_001"]
    )

    names = [variant.name for variant in variants]
    assert names == ["variant_002", "variant_003"]


def test_sample_strategy_variants_n_exceeds_combinations() -> None:
    sampling = {
        "portfolio.rank.n": {"dist": "categorical", "values": [1, 2]},
        "portfolio.weighting_scheme": {"dist": "discrete", "values": ["equal"]},
    }

    with pytest.raises(ValueError, match="exceeds available unique combinations"):
        sample_strategy_variants(sampling, 3, seed=5)


def test_sample_strategy_variants_n_exceeds_discrete_range_combinations() -> None:
    sampling = {
        "portfolio.rank.n": {"dist": "discrete", "low": 1, "high": 2, "step": 1},
        "portfolio.weighting_scheme": {"dist": "categorical", "values": ["equal"]},
    }

    with pytest.raises(ValueError, match="exceeds available unique combinations"):
        sample_strategy_variants(sampling, 3, seed=9)
