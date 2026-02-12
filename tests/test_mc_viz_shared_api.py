from __future__ import annotations

import inspect

from trend.mc import execute_mc_viz


def test_execute_mc_viz_public_signature() -> None:
    signature = inspect.signature(execute_mc_viz)
    assert list(signature.parameters) == [
        "bundle_path",
        "out_dir",
        "charts",
        "html",
        "json",
        "png",
    ]
    assert execute_mc_viz.__doc__
