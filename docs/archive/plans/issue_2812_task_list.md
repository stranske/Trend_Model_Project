# Issue #2812 – Actionlint Hardening Checklist

All acceptance criteria for [Issue #2812](https://github.com/stranske/Trend_Model_Project/issues/2812) have been implemented.

- [x] Switch the Actionlint workflow to fail on errors via reviewdog while preserving PR review annotations.
- [x] Cache the Actionlint binary on runners and reuse it across workflow invocations.
- [x] Load a curated Actionlint warning allowlist from source control before lint execution.
- [x] Replace inline heredoc scripts implicated in shellcheck parse warnings with dedicated helpers in `.github/scripts/`.
- [x] Document how to extend the Actionlint allowlist to keep future edits consistent.

Acceptance criteria verification:

- [x] Actionlint is configured to fail the build on errors and passes locally with the current workflows.
- [x] Updated workflows/scripts eliminate shellcheck heredoc parse errors, keeping CI annotations clean.

Manual verification:

- `actionlint` (with allowlist arguments from `.github/actionlint-allowlist.txt`)
