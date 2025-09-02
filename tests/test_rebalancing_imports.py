"""Test rebalancing module imports to prevent regression."""


def test_apply_rebalancing_strategies_import():
    """Test that apply_rebalancing_strategies can be imported from rebalancing module."""
    from trend_analysis.rebalancing import apply_rebalancing_strategies

    assert callable(apply_rebalancing_strategies)


def test_multi_period_engine_imports_rebalancing():
    """Test that multi_period engine can import apply_rebalancing_strategies."""
    # This is the exact import that was reported to fail

    # Verify the function is available in the engine module namespace
    import trend_analysis.multi_period.engine as engine

    assert hasattr(engine, "apply_rebalancing_strategies")
    assert callable(engine.apply_rebalancing_strategies)


def test_rebalancing_module_exports():
    """Test that all expected functions are exported from rebalancing module."""
    import trend_analysis.rebalancing as reb

    # Check that __all__ is defined and includes our function
    assert hasattr(reb, "__all__")
    assert "apply_rebalancing_strategies" in reb.__all__

    # Check that the function is accessible
    assert hasattr(reb, "apply_rebalancing_strategies")
    assert callable(reb.apply_rebalancing_strategies)


def test_apply_rebalancing_strategies_signature():
    """Test that apply_rebalancing_strategies has the expected signature."""
    from trend_analysis.rebalancing import apply_rebalancing_strategies
    import inspect

    sig = inspect.signature(apply_rebalancing_strategies)
    params = list(sig.parameters.keys())

    # Check required parameters
    assert "strategies" in params
    assert "strategy_params" in params
    assert "current_weights" in params
    assert "target_weights" in params

    # Check return type annotation
    assert sig.return_annotation is not inspect.Signature.empty
