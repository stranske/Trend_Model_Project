# Keepalive Status — PR #3791

> **Status:** In progress — keepalive checklist initialized; awaiting progress updates.

## Scope
- [ ] Leverage uv/pip caching to reuse compiled wheels and virtual environments across matrix entries and reruns.
- [ ] Trim redundant installs by reusing shared tool versions (from autofix pins) and avoiding duplicate pip install calls where uv sync already resolves dependencies.
- [ ] Non-goal: Changing test selection, markers, coverage thresholds, or adding dependencies beyond cache helpers.

## Tasks
- [ ] Enable uv cache (or actions/cache) for uv's cache directory and wheel artifacts keyed by Python version + lockfile, and document the cache keys.
- [ ] Consolidate formatter/test dependency installs to a single install path per run (e.g., rely on uv sync + pinned extras) to eliminate duplicate pip invocations.
- [ ] Add timing notes or a metrics hook to confirm install duration drops after caching.

## Acceptance criteria
- [ ] Python CI jobs reuse cached dependencies across matrix runs, with logs showing cache hits and reduced install time.
- [ ] Redundant install steps are removed/merged while still installing pinned tooling from .github/workflows/autofix-versions.env.
- [ ] CI remains green across reruns with the new cache strategy.
