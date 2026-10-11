# Issue 6080: Hypothesis is a development dependency

The source base was `f9761590f9db9e6c5f3b9be161d2b07490a45dc9`.
The repair was rebased onto ledger-only merge
`6b39e29bc817605441c65dc871096594f58397f9` before publication.

`hypothesis==6.155.7` moves from runtime dependencies to the `dev` extra.
The shared pin is unchanged. `rg -n hypothesis src` exits 1 with no matches;
production modules do not import Hypothesis. The dependency-enforcement test
requires both absence from runtime dependencies and the exact development pin.

## Deliberate RED and GREEN

With the new metadata assertion but the original pyproject, dependency enforcement
returned 1: one failed, three passed. The failed assertion was the runtime
Hypothesis prohibition. With the classification repaired, the requested focused
dependency tests returned 0: 15 passed, one skipped (optional tool availability).

The forced lockfile check also exposed a pre-existing mismatch: it compiled
runtime-only dependencies while the committed lock includes `app`, `dev`,
`notebooks`, and `llm`. That real check returned 1: one failed, three passed.
The repaired check uses those four extras and universal platform resolution,
matching `make lock`. It constrains resolution to the committed versions so a
fresh package-index release cannot turn consistency into an unrelated upgrade.
It still rejects incompatible pins, absent dependencies, and extra lock entries.
The command-scope regression checks the actual Makefile recipe as well.

Lock generation first used the requested four-extra command. On macOS it dropped
Linux-only packages, so the final lock was regenerated with `--universal`, as
the existing dependency-lock workflow already does. Existing versions remain
unchanged; platform markers and Windows-only colorama/tzdata are retained.
The forced check now returns 0: five passed. No skip flag was weakened.

## Reproducible receipt and limits

Raw focused test output and full SHA256/length bindings for the five changed
metadata/test files are retained in the owning lane's immutable receipt:
`20261011-6080-proof.json`, SHA256
`7c0aeea05c2c242a68fc9e999befa7373ae28b5b792f530dceccf6102fa9808c`.
Its harness reads the original pyproject directly from the full base commit and
demonstrates the same metadata predicate failing before and passing after.

Tests use the existing Trend Python 3.12 environment, with coverage/pytest-cov
from the existing isolated maintenance environment through PYTHONPATH. An earlier
system-Python attempt failed during incompatible NumPy/SciPy import; another
environment-only attempt lacked coverage. Neither is counted as product RED.
The merged ledger preflight was observed returning 0 in this local owned worker.
This is not a hosted Belt run, whole-repository test result, reviewer acceptance,
or authorization to modify/merge generated PR 6079. Maint 52 regeneration and
Maint 71 settlement remain separate post-merge campaign work.
