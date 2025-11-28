'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  extractScopeTasksAcceptanceSections,
  parseScopeTasksAcceptanceSections,
} = require('../issue_scope_parser');

test('extracts sections inside auto-status markers', () => {
  const issue = [
    'Intro text',
    '<!-- auto-status-summary:start -->',
    '## Scope',
    '- item a',
    '',
    '## Tasks',
    '- [ ] first',
    '',
    '## Acceptance Criteria',
    '- pass',
    '<!-- auto-status-summary:end -->',
    'Footer',
  ].join('\n');

  const result = extractScopeTasksAcceptanceSections(issue);
  assert.equal(
    result,
    ['#### Scope', '- item a', '', '#### Task List', '- [ ] first', '', '#### Acceptance Criteria', '- pass'].join('\n')
  );
});

test('parses plain headings without markdown hashes', () => {
  const issue = [
    'Issue Scope',
    '- summary',
    '',
    'Tasks:',
    '- [ ] alpha',
    '',
    'Acceptance criteria',
    '- ok',
  ].join('\n');

  const result = extractScopeTasksAcceptanceSections(issue);
  assert.equal(
    result,
    ['#### Scope', '- summary', '', '#### Task List', '- [ ] alpha', '', '#### Acceptance Criteria', '- ok'].join('\n')
  );
});

test('parseScopeTasksAcceptanceSections preserves structured sections', () => {
  const issue = [
    '## Issue Scope',
    '- overview line',
    '',
    '**Task List**',
    '- [ ] do one',
    '- [x] done two',
    '',
    'Acceptance criteria:',
    '- ✅ verified',
  ].join('\n');

  const parsed = parseScopeTasksAcceptanceSections(issue);
  assert.deepEqual(parsed, {
    scope: '- overview line',
    tasks: ['- [ ] do one', '- [x] done two'].join('\n'),
    acceptance: '- ✅ verified',
  });
});

test('returns empty string when no headings present', () => {
  const issue = 'No structured content here.';
  assert.equal(extractScopeTasksAcceptanceSections(issue), '');
});
