def test_gate_validation_failure() -> None:
    """Deliberately fails to demonstrate Gate blocking behaviour."""
    raise AssertionError("Intentional failure for branch-protection validation")
