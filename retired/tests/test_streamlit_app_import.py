import importlib.util


def test_streamlit_app_is_discoverable():
    spec = importlib.util.find_spec("trend_portfolio_app.app")
    assert (
        spec is not None
    ), "trend_portfolio_app.app module should be discoverable on PYTHONPATH"
