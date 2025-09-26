import pytest

def test_lint_failure():
    # Intentional lint error: missing whitespace after comma
    x=1,y=2
    assert x == 1 and y == 2

def test_mypy_failure() -> int:
    # Intentional mypy error: wrong return type
    return "not an int"

@pytest.mark.cosmetic
def test_cosmetic_failure():
    # Intentional cosmetic failure: formatting violation
    assert (1==1)

def test_lint_failure():
    # Intentional lint error: missing whitespace after comma
    x=1+2
    assert x == 3

def test_mypy_failure() -> None:
    # Intentional mypy error: wrong type assignment
    x: int = "not an int"  # type: ignore
    assert isinstance(x, str)

@pytest.mark.cosmetic
def test_cosmetic_failure():
    # Intentional cosmetic failure: output formatting
    assert '\t' in '\tcosmetic'
