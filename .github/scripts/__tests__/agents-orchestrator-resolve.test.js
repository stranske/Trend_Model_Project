'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const { resolveOrchestratorParams } = require('../agents_orchestrator_resolve');

function createSummary() {
  return {
    entries: [],
    addHeading(text) {
      this.entries.push({ type: 'heading', text });
      return this;
    },
    addTable(rows) {
      this.entries.push({ type: 'table', rows });
      return this;
    },
    addRaw(text) {
      this.entries.push({ type: 'raw', text });
      return this;
    },
    addEOL() {
      this.entries.push({ type: 'eol' });
      return this;
    },
    async write() {
      this.entries.push({ type: 'write' });
    }
  };
}

test('resolveOrchestratorParams merges configuration and summaries outputs', async () => {
  const outputs = {};
  const info = [];
  const warnings = [];
  const summary = createSummary();
  const core = {
    setOutput(key, value) {
      outputs[key] = value;
    },
    info(message) {
      info.push(message);
    },
    warning(message) {
      warnings.push(message);
    },
    summary
  };

  const labelError = new Error('not found');
  labelError.status = 404;

  const github = {
    rest: {
      issues: {
        async getLabel() {
          throw labelError;
        }
      }
    }
  };

  const env = {
    PARAMS_JSON: JSON.stringify({
      readiness_agents: ['copilot', 'codex', 'helper'],
      worker: { max_parallel: 3 },
      conveyor: { max_merges: 2 }
    }),
    WORKFLOW_DRY_RUN: 'true',
    WORKFLOW_KEEPALIVE_ENABLED: 'true',
    WORKFLOW_OPTIONS_JSON: JSON.stringify({
      belt: {
        dispatcher: { force_issue: '42' },
        worker: { max_parallel: 3 },
        conveyor: { max_merges: 2 }
      }
    })
  };

  await resolveOrchestratorParams({
    github,
    context: { repo: { owner: 'octo', repo: 'demo' } },
    core,
    env
  });

  assert.equal(outputs.readiness_agents, 'copilot,codex,helper');
  assert.equal(outputs.dispatcher_force_issue, '42');
  assert.equal(outputs.worker_max_parallel, '3');
  assert.equal(outputs.dry_run, 'true');
  assert.equal(outputs.enable_keepalive, 'true');
  assert.ok(summary.entries.length > 0);
  assert.ok(info.some((message) => message.includes('keepalive')));
  assert.equal(warnings.length, 0);
});
