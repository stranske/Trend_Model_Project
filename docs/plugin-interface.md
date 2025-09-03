# Plugin Interface: Selectors, Rebalancers, and Weight Engines

This document describes the lightweight plugin system used to register and create strategy components without modifying core code.

## Overview
- Registry: `PluginRegistry[T]` stores a mapping of plugin names to classes.
- Base classes: `Selector`, `Rebalancer`, `WeightEngine` in `src/trend_analysis/plugins/__init__.py`.
- Decorators: Use `@selector_registry.register("name")`, `@rebalancer_registry.register("name")`, and `@weight_engine_registry.register("name")` to register.
- Factories: `create_selector(name, **params)` and `create_rebalancer(name, params)` instantiate plugins by name.

## Add a new plugin
1. Create a file implementing the base class method(s).
2. Annotate the class with the appropriate registry decorator and a unique name.
3. Optionally expose a convenience factory.

Example (Selector):
```python
from trend_analysis.plugins import Selector, selector_registry
import pandas as pd

@selector_registry.register("top_n")
class TopNSelector(Selector):
    def __init__(self, top_n: int = 5, score_col: str = "score"):
        self.top_n = top_n
        self.score_col = score_col

    def select(self, score_frame: pd.DataFrame):
        """
        Select top N items from the DataFrame based on the score column.

        Parameters:
            score_frame (pd.DataFrame): DataFrame containing scores.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Selected rows and a log DataFrame with score column.
        """
        sel = score_frame.nlargest(self.top_n, self.score_col)
        log = sel[[self.score_col]].copy()
        return sel, log
```

## Discover available plugins
```python
from trend_analysis.plugins import selector_registry, rebalancer_registry, weight_engine_registry
print(selector_registry.available())
print(rebalancer_registry.available())
print(weight_engine_registry.available())
```

## Config usage
Strategies can be specified by name in config and constructed via the factories. Unknown names raise a clear `ValueError` listing available plugins.

## Tests
- Registry behavior and error handling are covered by tests (e.g., `tests/test_weight_engines.py`, `tests/test_rebalancing_*`, `tests/test_selector_weighting.py`).
