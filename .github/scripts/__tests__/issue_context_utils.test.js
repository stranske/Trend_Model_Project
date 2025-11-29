const test = require('node:test');
const assert = require('node:assert/strict');

const { buildIssueContext } = require('../issue_context_utils.js');

const COMPLETE_BODY = `
## Scope
- Evaluate automation bridge

## Tasks
- [ ] add invite metadata step

## Acceptance Criteria
- [ ] issue context posts on PR
`;

const INCOMPLETE_BODY = `
## Tasks
- [ ] missing other sections
`;

test('buildIssueContext returns summary without warnings when sections exist', () => {
  const result = buildIssueContext(COMPLETE_BODY);
  assert.equal(result.summaryNeedsWarning, false);
  assert.equal(result.warningLines.length, 0);
  assert.ok(result.statusSummaryBlock.includes('Automated Status Summary'));
  assert.ok(!result.statusSummaryBlock.includes('Summary Unavailable'));
  assert.ok(result.scopeBlock.includes('#### Scope'));
});

test('buildIssueContext flags warnings when sections missing', () => {
  const result = buildIssueContext(INCOMPLETE_BODY);
  assert.equal(result.summaryNeedsWarning, true);
  assert.ok(result.warningLines[0].includes('Template Warning'));
  assert.ok(result.statusSummaryBlock.includes('Summary Unavailable'));
  assert.ok(result.warningDetails.join('\n').includes('Scope'));
  assert.ok(result.warningDetails.join('\n').includes('Acceptance Criteria'));
});
