import importlib.util
import pathlib
import sys

if "ipyfilechooser" not in sys.modules:
    stub = type(sys)("ipyfilechooser")
    stub.FileChooser = object
    sys.modules["ipyfilechooser"] = stub

module_path = (
    pathlib.Path(__file__).resolve().parents[1]
    / "Old"
    / "Vol_Adj_Trend_Analysis_Cleanup.py"
)
spec = importlib.util.spec_from_file_location("cleanup", module_path)
cleanup = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cleanup)


def test_run_analysis_returns_none():
    assert (
        cleanup.run_analysis(None, None, None, None, None, None, None) is None
    )


def test_prepare_weights_missing():
    assert not hasattr(cleanup, "prepare_weights")
