from __future__ import annotations

from tests import conftest


def test_ci_does_not_force_a_single_xdist_worker() -> None:
    # ``-n auto`` must size the pool from the runner. Order-sensitive tests are
    # isolated with ``@pytest.mark.serial`` / ``xdist_group`` (honoured by the
    # Gate's ``--dist=loadgroup``) rather than by serialising the whole suite.
    assert not hasattr(conftest, "pytest_xdist_auto_num_workers")
