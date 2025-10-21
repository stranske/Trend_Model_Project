'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const test = require('node:test');
const assert = require('node:assert/strict');

const { normalizeCoverageArtifacts, runtimeFrom } = require('../coverage-normalize.js');

test('runtimeFrom strips coverage prefix', () => {
  assert.equal(runtimeFrom('coverage-3.11'), '3.11');
  assert.equal(runtimeFrom('3.12'), '3.12');
});

test('normalizeCoverageArtifacts computes stats and writes outputs', async () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'coverage-norm-'));
  const summaryDir = path.join(tmpDir, 'summary_artifacts');
  const coverage311Dir = path.join(summaryDir, 'coverage-runtimes', 'runtimes', '3.11');
  const coverage312Dir = path.join(summaryDir, 'coverage-runtimes', 'runtimes', '3.12');
  fs.mkdirSync(coverage311Dir, { recursive: true });
  fs.mkdirSync(coverage312Dir, { recursive: true });

  fs.writeFileSync(
    path.join(coverage311Dir, 'coverage.json'),
    JSON.stringify({ totals: { percent_covered: 91.234 } }),
  );
  fs.writeFileSync(
    path.join(coverage312Dir, 'coverage.xml'),
    '<coverage line-rate="0.88"></coverage>',
  );

  fs.mkdirSync(summaryDir, { recursive: true });
  fs.writeFileSync(
    path.join(summaryDir, 'coverage-trend.json'),
    JSON.stringify({ run_id: 200, run_number: 12, avg_coverage: 90.5, worst_job_coverage: 85.2 }),
  );
  fs.writeFileSync(
    path.join(summaryDir, 'coverage-trend-history.ndjson'),
    [
      JSON.stringify({ run_id: 100, run_number: 10, avg_coverage: 89.1, worst_job_coverage: 84.0 }),
      JSON.stringify({ run_id: 101, run_number: 11, avg_coverage: 90.0, worst_job_coverage: 84.5 }),
    ].join('\n'),
  );
  fs.writeFileSync(
    path.join(summaryDir, 'coverage-delta.json'),
    JSON.stringify({ current: 90.5, baseline: 92.0, delta: -1.5, status: 'drop' }),
  );

  const statsPath = path.join(tmpDir, 'coverage-stats.json');
  const deltaPath = path.join(tmpDir, 'coverage-delta-output.json');
  const outputs = {};
  const core = {
    setOutput: (key, value) => {
      outputs[key] = value;
    },
  };

  const result = await normalizeCoverageArtifacts({
    core,
    rootDir: summaryDir,
    statsPath,
    deltaPath,
  });

  assert.ok(fs.existsSync(statsPath));
  const stats = JSON.parse(fs.readFileSync(statsPath, 'utf8'));
  assert.equal(stats.job_coverages.length, 2);
  assert.ok(stats.coverage_table_markdown.includes('Runtime'));
  assert.ok(result.stats.avg_latest >= 88);
  assert.equal(result.delta.status, 'drop');
  assert.ok(outputs.stats_json);
  assert.ok(outputs.delta_json);
});
