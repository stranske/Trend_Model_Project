<!-- bootstrap for codex on issue #3647 -->

## Scope
- [x] Provide a durable breadcrumb for each Trend run that captures what ran, with what inputs, and when, alongside a simple one-page receipt for eyeballing results.

## Tasks
- [x] Emit a JSON manifest per run that records the config hash, git commit, data date range, instrument count, and essential metrics.
- [x] Generate a minimalist HTML report summarizing key metrics and linking to exported artifacts.
- [x] Save the manifest and HTML next to the exported outputs inside a timestamped run directory.

## Acceptance criteria
- [x] Each run produces both the manifest and HTML receipt that can be opened without other tooling.
- [x] Manifests contain sufficient metadata (config snapshot, hashes, run metadata) to reproduce the job.
