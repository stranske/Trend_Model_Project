import sys
import pathlib
import pytest
from hypothesis import given, strategies as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # noqa: E402

from trend_analysis import config  # noqa: E402

# Try to import ValidationError, fallback to ValueError for environments without pydantic
try:
    from pydantic import ValidationError
    ValidationException = ValidationError
except ImportError:
    ValidationException = ValueError

_DICT_SECTIONS = [
    "data",
    "preprocessing",
    "vol_adjust",
    "sample_split",
    "portfolio",
    "metrics",
    "export",
    "run",
]

invalid_values = st.one_of(
    st.integers(), st.floats(), st.booleans(), st.lists(st.integers())
)

_BASE_CFG = {"version": "1", "data": {}}


@given(field=st.sampled_from(_DICT_SECTIONS), val=invalid_values)
def test_sections_require_mappings(field, val):
    cfg = _BASE_CFG.copy()
    cfg[field] = val
    with pytest.raises((ValidationException, ValueError)):
        config.load_config(cfg)


@given(val=invalid_values)
def test_version_must_be_string(val):
    cfg = {"version": val, "data": {}}
    with pytest.raises((ValidationException, ValueError)):
        config.load_config(cfg)
