'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  extractScopeTasksAcceptanceSections,
  parseScopeTasksAcceptanceSections,
  analyzeSectionPresence,
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
    ['#### Scope', '- item a', '', '#### Tasks', '- [ ] first', '', '#### Acceptance Criteria', '- [ ] pass'].join('\n')
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
    ['#### Scope', '- summary', '', '#### Tasks', '- [ ] alpha', '', '#### Acceptance Criteria', '- [ ] ok'].join('\n')
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

test('parses blockquoted sections exported into PR bodies', () => {
  const issue = [
    '> ## Scope',
    '> ensure detection survives quoting',
    '>',
    '> ## Tasks',
    '> - [ ] first task',
    '> - [ ] second task',
    '>',
    '> ## Acceptance criteria',
    '> - two tasks completed',
  ].join('\n');

  const extracted = extractScopeTasksAcceptanceSections(issue);
  assert.equal(
    extracted,
    [
      '#### Scope',
      'ensure detection survives quoting',
      '',
      '#### Tasks',
      '- [ ] first task',
      '- [ ] second task',
      '',
      '#### Acceptance Criteria',
      '- [ ] two tasks completed',
    ].join('\n')
  );
});

test('returns empty string when no headings present', () => {
  const issue = 'No structured content here.';
  assert.equal(extractScopeTasksAcceptanceSections(issue), '');
});

test('includes placeholders when requested', () => {
  const issue = [
    'Tasks:',
    '- [ ] implement fast path',
  ].join('\n');

  const result = extractScopeTasksAcceptanceSections(issue, { includePlaceholders: true });
  assert.equal(
    result,
    [
      '#### Scope',
      '_No scope information provided_',
      '',
      '#### Tasks',
      '- [ ] implement fast path',
      '',
      '#### Acceptance Criteria',
      '- [ ] _No acceptance criteria defined_',
    ].join('\n')
  );
});

test('normalises bullet lists into checkboxes for tasks and acceptance', () => {
  const issue = [
    'Tasks',
    '- finish vectorisation',
    '-  add docs',
    '',
    'Acceptance criteria',
    '- confirm coverage > 90%',
    '-  ensure no regressions',
  ].join('\n');

  const result = extractScopeTasksAcceptanceSections(issue, { includePlaceholders: true });
  assert.equal(
    result,
    [
      '#### Scope',
      '_No scope information provided_',
      '',
      '#### Tasks',
      '- [ ] finish vectorisation',
      '- [ ] add docs',
      '',
      '#### Acceptance Criteria',
      '- [ ] confirm coverage > 90%',
      '- [ ] ensure no regressions',
    ].join('\n')
  );
});

test('analyzeSectionPresence flags missing sections', () => {
  const issue = [
    '## Scope',
    'ready to go',
    '',
    '## Tasks',
    '- [ ] build warning',
  ].join('\n');

  const status = analyzeSectionPresence(issue);
  assert.deepEqual(status.entries, [
    { key: 'scope', label: 'Scope', present: true },
    { key: 'tasks', label: 'Tasks', present: true },
    { key: 'acceptance', label: 'Acceptance Criteria', present: false },
  ]);
  assert.deepEqual(status.missing, ['Acceptance Criteria']);
  assert.equal(status.hasAllRequired, false);
});

test('analyzeSectionPresence recognises canonical template', () => {
  const issue = [
    '## Scope',
    '- new feature',
    '',
    '## Tasks',
    '- [ ] scaffold ui',
    '',
    '## Acceptance Criteria',
    '- [ ] demo recorded',
  ].join('\n');

  const status = analyzeSectionPresence(issue);
  assert.equal(status.hasAllRequired, true);
  assert.deepEqual(status.missing, []);
});
