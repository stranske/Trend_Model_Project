import sys

import pytest


def test_assert_deps_detects_explicitly_missing_modules(monkeypatch):
    from trend_analysis.proxy import server

    with monkeypatch.context() as m:
        for name in ["fastapi", "uvicorn", "httpx", "websockets"]:
            m.setitem(sys.modules, name, None)
        m.setattr(server, "_DEPS_AVAILABLE", True)

        with pytest.raises(ImportError) as excinfo:
            server._assert_deps()

    assert "Required dependencies not available" in str(excinfo.value)
