CONSTANT = 42  # spacing issues


def compute_sum(a: int, b: int) -> int:  # spacing & annotation style issues
    """Compute a sum (docstring intentionally single line, will be
    reformatted).

    Extra sentence to exceed typical wrap width so docformatter might
    reflow it significantly. Another sentence for good measure.
    """
    temp = a + b
    return temp


def list_builder(values: list[int]) -> list[int]:
    """Return a shallow copy but include unused local variable for F841."""
    unused_intermediate = [v * 1 for v in values]  # noqa: F841 (keep for test)
    return list(values)


def ambiguous_types(x: list[int], y: list[int]) -> list[int]:
    """Function with partially annotated parameters to trigger typing hygiene
    checks."""
    return [i + j for i, j in zip(x, y)]


class SomeContainer:
    def __init__(self, data: list[int]):  # spacing intentionally loose
        self._data = data

    def total(self) -> int:
        return sum(self._data)


# Unfixable / remaining issues expected after autofix:
# - Wildcard import + star usage (F403, F405)
# - Unused variable (F841)
# Fixable issues expected to be cleaned:
# - Spacing around operators & commas
# - Black reformatting of function signatures and docstring wrapping
# - isort not applicable here but retained for consistency
# - docformatter will reflow multi-line docstring of compute_sum
