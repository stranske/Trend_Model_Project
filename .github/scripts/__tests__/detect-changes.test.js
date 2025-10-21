'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const { detectChanges, isDocFile, isDockerRelated } = require('../detect-changes.js');

test('classifies documentation files using extensions, directories, and basenames', () => {
  assert.ok(isDocFile('README.md'));
  assert.ok(isDocFile('docs/guide/intro.rst'));
  assert.ok(isDocFile('Guidelines/STYLE.MD'));
  assert.ok(isDocFile('handbook/overview.txt'));
  assert.ok(isDocFile('assets/docs/diagram.svg')); // suffix with docs folder
  assert.ok(!isDocFile('src/index.js'));
  assert.ok(!isDocFile('package.json'));
});

test('detects docker-related files by path and filename', () => {
  assert.ok(isDockerRelated('Dockerfile'));
  assert.ok(isDockerRelated('services/api/Dockerfile.dev'));
  assert.ok(isDockerRelated('docker/Dockerfile'));
  assert.ok(isDockerRelated('.dockerignore'));
  assert.ok(isDockerRelated('ops/.docker/config.yaml'));
  assert.ok(!isDockerRelated('src/docker.ts'));
  assert.ok(!isDockerRelated('docs/docker.md'));
});

test('detectChanges reports doc-only fast-pass results', async () => {
  const outputs = {};
  const core = {
    setOutput: (key, value) => {
      outputs[key] = value;
    },
  };
  const context = {
    eventName: 'pull_request',
    repo: { owner: 'octo', repo: 'example' },
    payload: { pull_request: { number: 42 } },
  };
  const github = {
    rest: { pulls: { listFiles: () => {} } },
    paginate: async (method, params) => {
      assert.strictEqual(method, github.rest.pulls.listFiles);
      assert.equal(params.pull_number, 42);
      return [
        { filename: 'docs/usage.md' },
        { filename: 'guides/intro/README.mdx' },
      ];
    },
  };

  const result = await detectChanges({ github, context, core });
  assert.equal(result.docOnly, true);
  assert.equal(outputs.doc_only, 'true');
  assert.equal(outputs.run_core, 'false');
  assert.equal(outputs.reason, 'docs_only');
  assert.equal(outputs.docker_changed, 'false');
});

test('non pull request events set defaults', async () => {
  const outputs = {};
  const core = {
    setOutput: (key, value) => {
      outputs[key] = value;
    },
  };
  const context = {
    eventName: 'workflow_dispatch',
  };
  const github = {
    paginate: async () => {
      throw new Error('Should not be called for non PR events');
    },
  };

  const result = await detectChanges({ github, context, core });
  assert.equal(result.docOnly, false);
  assert.equal(result.runCore, true);
  assert.equal(outputs.reason, 'non_pr_event');
  assert.equal(outputs.docker_changed, 'true');
});
