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
The repaired check uses those four extras and explicit Linux platform resolution,
matching `make lock`. It constrains resolution to the committed versions so a
fresh package-index release cannot turn consistency into an unrelated upgrade.
It still rejects incompatible pins, absent dependencies, and extra lock entries.
The command-scope regression checks the actual Makefile recipe as well.

Lock generation first used the requested four-extra command. On macOS it dropped
Linux-only packages. An initial universal lock restored them but introduced a
valid reviewer P1: Docker's freeze comparison and `trend check` are not marker
aware. That expansion is withdrawn instead of changing deployment parsers in
this dependency repair. `make lock` and the forced check now explicitly target
Linux, preserving the existing concrete, marker-free lock consumer contract.
Every original package/version remains unchanged; only the command header and
resolver provenance comments differ. The final forced check returns 0: seven
passed, including a marker-free deployment-format regression. That regression
deliberately failed against reviewed head
`11f4375afe12a77755e399cb544e304c6a76f04c` before the correction. No skip flag was
weakened. A second reviewer counterexample identified the repository-local
Dependabot lock writer still using universal output. Both its check and write
commands now use the same four-extra Linux scope. The new automated-generation
regression failed against that existing workflow before this correction, then
passed. This workflow path is absent from the Workflows sync manifest, so the
change is repo-local, not an ad hoc edit of a centrally managed file.
The Docker image itself has not been built locally.

## Reproducible receipt and limits

Raw focused test output and full SHA256/length bindings for the six changed
metadata/test files are retained in the owning lane's immutable receipt:
`20261011-6080-proof-v3.json`, SHA256
`6201a5abcdbf59407885115b46fba246f52db76b3e4c1bc65eff476f802c1a78`.
Its harness reads the original pyproject directly from the full base commit and
demonstrates the same metadata predicate failing before and passing after,
plus the reviewed/fixed marker-format RED/GREEN. The original v1/v2 receipts are
retained as history, not reused to attest the corrected files.

Tests use the existing Trend Python 3.12 environment, with coverage/pytest-cov
from the existing isolated maintenance environment through PYTHONPATH. An earlier
system-Python attempt failed during incompatible NumPy/SciPy import; another
environment-only attempt lacked coverage. Neither is counted as product RED.
The merged ledger preflight was observed returning 0 in this local owned worker.
This is not a hosted Belt run, whole-repository test result, reviewer acceptance,
or authorization to modify/merge generated PR 6079. Maint 52 regeneration and
Maint 71 settlement remain separate post-merge campaign work.
