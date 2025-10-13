def test_gate_validation_failure() -> None:
    """Intentional failure to demonstrate Gate branch protection."""
    raise AssertionError("Intentional failure for Gate validation")
