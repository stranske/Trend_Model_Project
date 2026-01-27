# Tag filtering

## Monte Carlo scenario registry

The Monte Carlo scenario registry supports optional tag filtering via
`list_scenarios(tags=[...])`.

### Matching logic

- Matching is **case-insensitive**.
- Leading/trailing whitespace is ignored.
- A scenario is included when **any** supplied tag matches **any** tag on the
  scenario (logical OR).

### Example

Given a registry entry:

```yaml
scenarios:
  - name: hf_equity_ls_10y
    tags: [equity, hedge_fund, production]
```

`list_scenarios(tags=["HEDGE_FUND", "macro"])` returns the scenario because
`hedge_fund` matches one of the provided tags.
