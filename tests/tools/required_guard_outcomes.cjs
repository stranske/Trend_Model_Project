// Run against an authenticated Workflows source checkout; never copy its classifier.
// node tests/tools/required_guard_outcomes.cjs /absolute/Workflows/.github/scripts/sync_pr_merge_contract.js
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const classifierPath = process.argv[2];
assert.ok(classifierPath && path.isAbsolute(classifierPath), 'Provide the absolute source classifier path');
const { classifySyncPrChecks } = require(classifierPath);
const { required_contexts: declaredContexts } = JSON.parse(fs.readFileSync(
  path.resolve(__dirname, '../../.github/config/required-contexts.json'), 'utf8',
));
const headStatus = 'Health 45 Agents Guard / guard';
assert.deepEqual(declaredContexts, ['Gate / gate', 'guard', headStatus]);
const check = (name, conclusion) => ({ name, status: 'completed', conclusion, app: { slug: 'github-actions' } });
const classify = (jobConclusion, statusConclusion) => classifySyncPrChecks({
  checkRuns: [check('Gate / gate', 'success'), check('guard', jobConclusion), check(headStatus, statusConclusion)],
  requiredContexts: ['Gate / gate'], declaredContexts,
});
assert.equal(classify('success', 'success').status, 'ready');
for (const conclusion of ['failure', 'cancelled', 'timed_out', 'action_required']) {
  for (const failingContext of ['guard', headStatus]) {
    const result = classify(failingContext === 'guard' ? conclusion : 'success',
      failingContext === headStatus ? conclusion : 'success');
    assert.equal(result.status, 'checks_failed', `${failingContext}: ${conclusion}`);
    assert.deepEqual(result.declared_contexts.failed, [failingContext]);
  }
}
console.log('PASS: 9 actual source-classifier cases; passing Gate cannot mask either failed guard result');
