import pandas as pd

from trend_analysis.core import rank_selection


def test_canonicalise_labels_trims_and_deduplicates() -> None:
    labels = [" alpha ", "", "Alpha", "beta", "", "beta"]

    clean = rank_selection._canonicalise_labels(labels)

    assert clean == ["alpha", "Unnamed_2", "Alpha", "beta", "Unnamed_5", "beta_2"]


def test_ensure_canonical_columns_preserves_reference_when_clean() -> None:
    frame = pd.DataFrame([[1, 2]], columns=["a", "b"])

    canonical = rank_selection._ensure_canonical_columns(frame)

    assert canonical is frame
    pd.testing.assert_frame_equal(canonical, frame)


def test_hash_universe_stable_sorting() -> None:
    universe_one = ["BBB", "AAA", "CCC"]
    universe_two = ["CCC", "BBB", "AAA"]

    assert rank_selection._hash_universe(universe_one) == rank_selection._hash_universe(
        universe_two
    )
