'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  extractScopeTasksAcceptanceSections,
  findScopeTasksAcceptanceBlock,
} = require('../../../scripts/keepalive-runner.js');

test('extractScopeTasksAcceptanceSections accepts varied heading styles', () => {
  const block = [
    '<!-- auto-status-summary:start -->',
    '## Automated Status Summary',
    '## Scope',
    '- [ ] alpha',
    '### Tasks',
    '- [ ] beta',
    '#### Acceptance criteria',
    '- [ ] gamma',
    '<!-- auto-status-summary:end -->',
  ].join('\n');

  const extracted = extractScopeTasksAcceptanceSections(block);
  const expected = [
    '#### Scope',
    '- [ ] alpha',
    '',
    '#### Tasks',
    '- [ ] beta',
    '',
    '#### Acceptance Criteria',
    '- [ ] gamma',
  ].join('\n');

  assert.equal(extracted, expected);
});

test('findScopeTasksAcceptanceBlock falls back to bold headings in PR body', () => {
  const prBody = [
    '**Scope**',
    '- [ ] keep the UI optional',
    '',
    '**Tasks**',
    '- [ ] gate heavy imports behind availability checks',
    '',
    '**Acceptance criteria**',
    '- [ ] pipeline executes without widget dependencies',
  ].join('\n');

  const extracted = findScopeTasksAcceptanceBlock({ prBody, comments: [], override: '' });
  const expected = [
    '#### Scope',
    '- [ ] keep the UI optional',
    '',
    '#### Tasks',
    '- [ ] gate heavy imports behind availability checks',
    '',
    '#### Acceptance Criteria',
    '- [ ] pipeline executes without widget dependencies',
  ].join('\n');

  assert.equal(extracted, expected);
});

test('findScopeTasksAcceptanceBlock accepts plain headings with colons', () => {
  const prBody = [
    'Scope:',
    '- [ ] headline summary',
    '',
    'Tasks',
    '- [ ] do the actual implementation',
    '',
    'Acceptance criteria',
    '- [ ] passes the regression suite',
  ].join('\n');

  const extracted = findScopeTasksAcceptanceBlock({ prBody, comments: [], override: '' });
  const expected = [
    '#### Scope',
    '- [ ] headline summary',
    '',
    '#### Tasks',
    '- [ ] do the actual implementation',
    '',
    '#### Acceptance Criteria',
    '- [ ] passes the regression suite',
  ].join('\n');

  assert.equal(extracted, expected);
});
