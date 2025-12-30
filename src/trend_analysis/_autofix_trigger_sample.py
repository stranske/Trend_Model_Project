#  This file intentionally violates style conventions for test of autofix workflows.
#  Issues included:
#  - extra spaces in imports
#  - wildcard import usage
#  - inconsistent indentation
#  - double spaces
#  - badly formatted function definitions
#  - trailing whitespace
#  - long line exceeding typical 88 char limit
#  - unused variables


def badly_formatted_function(x: int, y: int = 5) -> int:
    temp = x + y
    return temp


def another_func(a: list[int], b: list[int]) -> list[int]:
    result = [i + j for i, j in zip(a, b)]
    return result


class Demo:
    def method(self, value: float) -> float:
        return value * 2.0


CONSTANT_VALUE = 42  # unused constant for test


def long_line() -> str:
    # Intentionally verbose string broken across lines to satisfy E501 while preserving content.
    return (
        "This is an intentionally overly verbose string whose primary purpose "
        "is to exceed the standard enforced line length so that formatting "
        "tools act."
    )
