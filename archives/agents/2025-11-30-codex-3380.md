<!-- bootstrap for codex on issue #3380 -->

## Scope
- [ ] Ship the new `rank` manager-selection mode alongside existing modes without regressing current defaults.
- [ ] Keep the rank workflow fully config-driven and vectorised, including demo and debug scripts.
- [ ] Expose the rank-selection controls in the ipywidgets UI flow and export the selected managers through existing exporters.

## Tasks
- [ ] Extend the selection config schema and sample YAML to cover `rank` (including `inclusion_approach`, `score_by`, and thresholds).
- [ ] Implement or update pure selection helpers so `rank_select_funds` honours sorting direction, blended weights, and inclusion rules.
- [ ] Update demo/debug scripts and documentation so the new mode is exercised end-to-end (including exporter outputs).
- [ ] Add UI wiring for mode/vol-adjust controls, rank-parameter widgets, and manual override integration.
- [ ] Backfill or adjust unit tests to cover ranking logic (ascending vs. descending metrics, blended weights, threshold handling).

## Acceptance criteria
- [ ] Config-driven runs (including demos) support `mode: rank` with top-N, top-% and threshold inclusion and respect metric direction.
- [ ] ipywidgets flow exposes the new rank controls, manual override still functions, and exports succeed for all configured formats.
- [ ] Ranking helpers are fully vectorised, normalise blended weights, and include unit tests for positive and negative merit metrics.
- [ ] Documentation and examples (including sample YAML and debug guidance) explain how to configure and validate the rank mode.
