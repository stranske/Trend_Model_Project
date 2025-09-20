from typing import *  # noqa: F401,F403


# Intentional violations for clean autofix scenario
def bad_func(a: int, b: int, c=[]):
    x = 1 + 2
    return x
