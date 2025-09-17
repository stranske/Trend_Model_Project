# Contributing to Trend Model Project

Thank you for your interest in improving the Trend Model Project! This document
explains how to work with the repository now that `main` is the default branch
and branch protections enforce CI, Docker, and release-note quality gates.

## Branching and Pull Requests

- Fork or create a feature branch from `main`.
- Keep your branch up to date with `main` by rebasing or merging regularly.
- All pull requests **must target `main`**. Automation will reject PRs aimed at
  other branches.
- Each pull request requires at least one approving code-owner review and GitHub
  auto-merge is configured for squash merges only.

## Required Checks Before Merge

Branch protection rules block merges until the following checks succeed:

1. **CI** – `gate / all-required-green` from `.github/workflows/ci.yml`.
2. **Docker** – `Docker / build` from `.github/workflows/docker.yml`.
3. **Release Notes** – `Release Notes / generate`, which validates that
   `git-cliff` can build the unreleased changelog.

GitHub enforces these checks automatically, but contributors should run local
validation before pushing to ensure quick feedback.

### Local Test Matrix

```bash
# Set up environment (once)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"

# Run the project test wrapper
./scripts/run_tests.sh
```

The script executes the same pytest subset as the CI workflow. Fix any failures
before sending your pull request.

### Release Notes Dry Run

The repository uses [`git-cliff`](https://git-cliff.org/) with the configuration
stored in `cliff.toml`. Generate the unreleased changelog locally to catch
format issues before the workflow runs:

```bash
# Ensure full git history so git-cliff can compute the changelog
git fetch origin --tags

# Build unreleased notes to release-notes.md
git-cliff --unreleased --strip header --output release-notes.md
```

Include the resulting preview in your manual testing notes or attach it to the
PR if reviewers request context.

## Documentation Updates

- Update `docs/ops/codex-bootstrap-facts.md` if operational facts change (e.g.,
  default branch, required checks).
- Reference `CONTRIBUTING.md` in new documentation so contributors find the
  governance rules quickly.

## Questions

If you are unsure about process changes or the governance model, open a
discussion or ping the maintainers via an issue before raising a pull request.
