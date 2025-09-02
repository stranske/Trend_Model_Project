# pytest conftest.py - configuration automatically handled via PYTHONPATH
#
# NOTE: Dependencies like NumPy may attempt to set PYTHONHASHSEED during test execution
# via monkeypatch.setenv('PYTHONHASHSEED', '0'). This has no effect since PYTHONHASHSEED
# must be set before the Python interpreter starts. For reproducible hash behavior,
# set PYTHONHASHSEED=0 in the environment before running Python/pytest.
