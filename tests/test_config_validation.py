import sys
import pathlib
import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # noqa: E402

from trend_analysis import config  # noqa: E402

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


@given(field=st.sampled_from(_DICT_SECTIONS), val=invalid_values)
def test_sections_require_mappings(field, val):
    with pytest.raises(ValidationError):
        config.load({field: val})


@given(val=invalid_values)
def test_version_must_be_string(val):
    with pytest.raises(ValidationError):
        config.load({"version": val})
