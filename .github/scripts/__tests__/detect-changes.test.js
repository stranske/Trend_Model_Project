'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { analyseChangedFiles, classifyDocument, isDockerRelated } = require('../detect-changes');

test('classifies documentation files', () => {
  assert.equal(classifyDocument('docs/guide.md'), true);
  assert.equal(classifyDocument('README.MD'), true);
  assert.equal(classifyDocument('src/app.py'), false);
});

test('detects docker-related paths', () => {
  assert.equal(isDockerRelated('Dockerfile'), true);
  assert.equal(isDockerRelated('infrastructure/docker/docker-compose.yml'), true);
  assert.equal(isDockerRelated('src/module.py'), false);
});

test('computes docs-only change summary', () => {
  const result = analyseChangedFiles(['docs/guide.md', 'README.md']);
  assert.equal(result.docOnly, true);
  assert.equal(result.runCore, false);
  assert.equal(result.reason, 'docs_only');
  assert.equal(result.dockerChanged, false);
});

test('computes mixed changes summary', () => {
  const result = analyseChangedFiles(['docs/guide.md', 'src/app.py']);
  assert.equal(result.docOnly, false);
  assert.equal(result.runCore, true);
  assert.equal(result.reason, 'code_changes');
});

test('detects docker change reason', () => {
  const result = analyseChangedFiles(['src/app.py', 'docker/Dockerfile']);
  assert.equal(result.dockerChanged, true);
  assert.equal(result.reason, 'code_changes');
});

test('handles empty change list', () => {
  const result = analyseChangedFiles([]);
  assert.equal(result.docOnly, true);
  assert.equal(result.runCore, false);
  assert.equal(result.reason, 'no_changes');
});
