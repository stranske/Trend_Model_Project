'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { summarizeArtifacts } = require('../coverage-normalize');

test('summarizes successful scenario', () => {
  const result = summarizeArtifacts({
    artifactNames: ['sf-minimal-coverage-3.11'],
    jobs: [{ name: 'Scenario - minimal', conclusion: 'success' }],
    scenarioNames: ['minimal'],
    pythonVersions: ['3.11'],
    runId: 123,
  });
  assert.equal(result.failures, 0);
  assert.ok(result.table.includes('minimal'));
  assert.equal(result.report.failures, 0);
  assert.equal(result.report.artifact_count, 1);
});

test('detects missing artifacts', () => {
  const result = summarizeArtifacts({
    artifactNames: [],
    jobs: [{ name: 'Scenario - minimal', conclusion: 'success' }],
    scenarioNames: ['minimal'],
    pythonVersions: ['3.11'],
    runId: 456,
  });
  assert.equal(result.failures, 1);
  assert.ok(result.table.includes('sf-minimal-coverage-3.11'));
  assert.equal(result.report.failures, 1);
});

test('warns on unknown scenario', () => {
  const warnings = [];
  const result = summarizeArtifacts({
    artifactNames: [],
    jobs: [],
    scenarioNames: ['mystery'],
    pythonVersions: ['3.11'],
    runId: 789,
    warn: (message) => warnings.push(message),
  });
  assert.equal(warnings.length, 2);
  assert.equal(result.failures, 1);
  assert.ok(result.table.includes('mystery'));
});
