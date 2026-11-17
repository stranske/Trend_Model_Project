# Deliberate-break evidence: in-sample regime proxy (#6229 / PR #6233)

Linked issue: `stranske/Trend_Model_Project#6229` (single-period MC regime turnover caps).

## Mutation (temporary, not committed)

In `MonteCarloRunner._resolve_turnover_cap_for_context`, `in_sample_returns` was set to
`context.returns` instead of `self._in_sample_returns(context.returns, split_cfg)`, restoring
pre-fix full-path proxy resolution (future OOS observations affect the cap).

## RED — full-path proxy, named causality test

`python3 -m pytest tests/monte_carlo/test_runner_turnover_diagnostics.py::test_single_period_cap_ignores_out_of_sample_proxy -q -m "not slow"`

```
FAILED tests/monte_carlo/test_runner_turnover_diagnostics.py::test_single_period_cap_ignores_out_of_sample_proxy
...
E       assert 0.05 == 0.8 ± 8.0e-07
```

Paths with identical in-sample calm data but opposing OOS stress tails incorrectly select the
`stress` cap (`0.05`) instead of the shared `calm` cap (`0.80`).

## GREEN — in-sample slice restored, same command

```
1 passed in 0.72s
```

Recorded at branch tip during keepalive verification for PR #6233.