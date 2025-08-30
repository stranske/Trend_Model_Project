"""Test to ensure no circular import issues remain after architectural cleanup."""

import pytest


def test_no_importlib_workarounds_in_sim_runner():
    """Test that sim_runner no longer uses importlib workarounds for trend_analysis."""
    from trend_portfolio_app import sim_runner
    import inspect
    
    # Check that we don't have the defensive importlib pattern
    source = inspect.getsource(sim_runner)
    assert "importlib.import_module" not in source
    assert "HAS_TA" not in source
    assert "ta_pipeline" not in source
    
    # Ensure we have proper imports instead
    assert "from trend_analysis.pipeline import single_period_run" in source
    assert "from trend_analysis.core.rank_selection import RiskStatsConfig" in source


def test_sim_runner_uses_trend_analysis_directly():
    """Test that sim_runner imports and uses trend_analysis components directly."""
    from trend_portfolio_app.sim_runner import compute_score_frame
    from trend_analysis.pipeline import single_period_run
    import pandas as pd
    
    # Create test data
    df = pd.DataFrame(
        {'A': [0.01, 0.02, -0.01], 'B': [0.02, -0.01, 0.01]}, 
        index=pd.date_range('2023-01-01', periods=3, freq='ME')
    )
    
    # Test that compute_score_frame works with our data
    result = compute_score_frame(df, df.index[0], df.index[-1])
    
    # Should return a proper DataFrame with metrics
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Two assets
    assert len(result.columns) >= 6  # Should have standard metrics
    
    # Verify it has expected metrics columns
    expected_columns = {'AnnualReturn', 'Volatility', 'Sharpe', 'Sortino', 'MaxDrawdown', 'InformationRatio'}
    assert expected_columns.issubset(set(result.columns))


def test_all_key_imports_work_cleanly():
    """Test that all key modules can be imported without circular dependency issues."""
    # These should all work without any ImportError or circular import issues
    import trend_analysis
    import trend_analysis.pipeline
    import trend_analysis.multi_period
    import trend_analysis.rebalancing
    import trend_portfolio_app
    import trend_portfolio_app.sim_runner
    
    # Test specific functions can be imported
    from trend_analysis.rebalancing import apply_rebalancing_strategies
    from trend_analysis.multi_period.engine import run
    from trend_portfolio_app.sim_runner import Simulator
    
    # All should be callable
    assert callable(apply_rebalancing_strategies)
    assert callable(run)
    assert callable(Simulator)


def test_config_package_has_clean_imports():
    """Test that config package doesn't use complex importlib workarounds."""
    from trend_analysis import config
    import inspect
    
    # Check the config package init
    source = inspect.getsource(config)
    
    # Should not have complex importlib workarounds
    assert "importlib.util.spec_from_file_location" not in source
    assert "_import_config_module" not in source
    
    # Should use normal imports from models
    assert "from .models import" in source


def test_proper_dependency_structure():
    """Test that the dependency structure is clean without defensive imports."""
    # trend_portfolio_app should cleanly depend on trend_analysis
    from trend_portfolio_app.sim_runner import compute_score_frame
    from trend_analysis.pipeline import single_period_run
    
    # They should be the same function or compatible
    # We can't test equality, but we can test they both exist and are callable
    assert callable(compute_score_frame)
    assert callable(single_period_run)
    
    # Test that compute_score_frame actually uses single_period_run internally
    # by checking it produces consistent results
    import pandas as pd
    
    # Create test data with Date column as expected by single_period_run
    df_with_date = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=3, freq='ME'), 
        'A': [0.01, 0.02, -0.01], 
        'B': [0.02, -0.01, 0.01]
    })
    
    df_indexed = df_with_date.set_index('Date')
    
    # Both should work and produce similar structures
    result_sim = compute_score_frame(df_indexed, df_indexed.index[0], df_indexed.index[-1])
    result_direct = single_period_run(df_with_date, '2023-01', '2023-03')
    
    assert isinstance(result_sim, pd.DataFrame)
    assert isinstance(result_direct, pd.DataFrame)
    assert len(result_sim) == len(result_direct)  # Same number of assets