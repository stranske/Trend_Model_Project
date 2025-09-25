from __future__ import annotations

from typing import Any  # isort: skip
import numpy as np  # isort: skip

"""Central NumPy typing aliases (float64 focused).

Concentrates array generics so modules avoid verbose ``np.ndarray`` parameter
expressions.  Keep minimal; extend only when additional dtypes become common.
"""

_F64 = np.float64

FloatArray = np.ndarray[Any, np.dtype[_F64]]
AnyArray = np.ndarray[Any, Any]

__all__ = ["FloatArray", "AnyArray"]
