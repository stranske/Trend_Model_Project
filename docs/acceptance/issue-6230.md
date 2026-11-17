# Acceptance record: stranske/Trend_Model_Project#6230

Monte Carlo `terminal_wealth` and `time_underwater` path metrics.

Checked 2026-11-17 UTC on branch `claude/issue-6230` at
`ce898fdd244bf838b0a3e7c5d20197470a7f54d4`, Python 3.14.10.

## Named test passes

```bash
python3 -m pytest tests/monte_carlo/test_results.py::test_results_frame_includes_terminal_wealth_and_time_underwater -q
```

Result: `1 passed`.

The test runs `MonteCarloRunner._evaluate_strategy` on two user-weighted return
paths and asserts on `build_results_frame`:

- drawdown + recovery NAV (normalised 1.0, 1.10, 0.88, 0.924, 1.155, 1.1781)
  gives `time_underwater == 2` and `terminal_wealth == 1.1781`, which equals
  `nav.iloc[-1] / nav.iloc[0]`;
- monotone NAV gives `time_underwater == 0` and `terminal_wealth == 1.01**5`.

## Deliberate break, then revert

Break: stop adding the metrics to `evaluation.metrics` in
`src/trend_analysis/monte_carlo/runner.py:_evaluate_strategy` (line 808):

```diff
         if nav_series is not None:
             metrics = dict(metrics)
-            metrics.update(self._nav_path_metrics(nav_series))
+            pass  # DELIBERATE BREAK: metrics.update(self._nav_path_metrics(nav_series))
```

Command (same as above) with the break applied. Result: **`1 failed`**, because
the columns are missing from the results frame:

```text
>       assert {"terminal_wealth", "time_underwater"} <= set(frame.columns)
E       AssertionError: assert {'terminal_we...e_underwater'} <= {'cagr', 'fol...path_id', ...}
tests/monte_carlo/test_results.py:623: AssertionError
FAILED tests/monte_carlo/test_results.py::test_results_frame_includes_terminal_wealth_and_time_underwater
```

Revert: `git checkout -- src/trend_analysis/monte_carlo/runner.py` (working tree
clean for `src/` afterwards). Same command again. Result: **`1 passed`**.

## Whole test file passes

```bash
python3 -m pytest tests/monte_carlo/test_results.py -q
```

Result: `27 passed`.

Refs #6230
