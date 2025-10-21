'use strict';

const fs = require('fs');
const path = require('path');

function toParts(value) {
  return path
    .normalize(value)
    .split(path.sep)
    .filter((part) => part && part !== '.' && part !== path.sep);
}

function findFirst(root, targetName) {
  if (!fs.existsSync(root)) {
    return null;
  }
  const stack = [root];
  while (stack.length) {
    const current = stack.pop();
    const stat = fs.statSync(current);
    if (stat.isDirectory()) {
      const entries = fs.readdirSync(current);
      for (const entry of entries) {
        stack.push(path.join(current, entry));
      }
    } else if (stat.isFile()) {
      if (path.basename(current) === targetName) {
        return current;
      }
    }
  }
  return null;
}

function readCoverageFromXml(xmlPath) {
  try {
    const content = fs.readFileSync(xmlPath, 'utf8');
    const match = content.match(/line-rate\s*=\s*"([0-9.]+)"/);
    if (match) {
      const value = Number.parseFloat(match[1]);
      if (Number.isFinite(value)) {
        return value * 100;
      }
    }
  } catch (error) {
    // ignore parse errors
  }
  return null;
}

function safeNumber(value) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function readCoverageFromJson(jsonPath) {
  try {
    const data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
    if (data && typeof data === 'object') {
      const totals = data.totals;
      if (totals && typeof totals === 'object') {
        if (typeof totals.percent_covered === 'number') {
          return totals.percent_covered;
        }
        if (typeof totals.percent_covered_display === 'string') {
          const value = safeNumber(totals.percent_covered_display);
          if (value !== null) {
            return value;
          }
        }
        const coveredCandidates = [
          totals.covered_lines,
          totals.covered_statements,
          totals.covered,
        ];
        const totalCandidates = [
          totals.num_statements,
          totals.num_lines,
          totals.statements,
        ];
        const coveredValue = coveredCandidates.find((candidate) => safeNumber(candidate) !== null);
        const totalValue = totalCandidates.find((candidate) => safeNumber(candidate) !== null);
        const covered = safeNumber(coveredValue);
        const total = safeNumber(totalValue);
        if (total === 0) {
          return 0;
        }
        if (covered !== null && total !== null && total > 0) {
          return (covered / total) * 100;
        }
      }
    }
  } catch (error) {
    // ignore parse errors
  }
  return null;
}

function readCoverage(directory) {
  const xmlPath = path.join(directory, 'coverage.xml');
  if (fs.existsSync(xmlPath)) {
    const value = readCoverageFromXml(xmlPath);
    if (value !== null) {
      return value;
    }
  }

  const jsonPath = path.join(directory, 'coverage.json');
  if (fs.existsSync(jsonPath)) {
    const value = readCoverageFromJson(jsonPath);
    if (value !== null) {
      return value;
    }
  }
  return null;
}

function labelFor(directory) {
  const parts = toParts(directory);
  for (let i = 0; i < parts.length - 1; i += 1) {
    if (parts[i] === 'runtimes') {
      return `coverage-${parts[i + 1]}`;
    }
  }
  for (let i = parts.length - 1; i >= 0; i -= 1) {
    if (parts[i].startsWith('coverage-')) {
      return parts[i];
    }
  }
  if (parts.length) {
    return `coverage-${parts[parts.length - 1]}`;
  }
  return null;
}

function discoverPayloads(base) {
  if (!fs.existsSync(base) || !fs.statSync(base).isDirectory()) {
    return [];
  }
  const discovered = [];
  const seen = new Set();

  const stack = [base];
  while (stack.length) {
    const current = stack.pop();
    const stat = fs.statSync(current);
    if (stat.isDirectory()) {
      const entries = fs.readdirSync(current, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(current, entry.name);
        if (entry.isDirectory()) {
          stack.push(fullPath);
        } else if (entry.isFile()) {
          if (entry.name === 'coverage.xml' || entry.name === 'coverage.json') {
            const directory = path.dirname(fullPath);
            const key = path.resolve(directory);
            if (!seen.has(key)) {
              const label = labelFor(directory);
              if (label) {
                discovered.push([label, directory]);
                seen.add(key);
              }
            }
          }
        }
      }
    }
  }
  return discovered;
}

function runtimeFrom(name) {
  const prefix = 'coverage-';
  return name.startsWith(prefix) ? name.slice(prefix.length) : name;
}

function naturalSortKey(name) {
  const runtime = runtimeFrom(name);
  const segments = runtime.split(/(\d+)/);
  const key = [];
  for (const segment of segments) {
    if (!segment) {
      continue;
    }
    if (/^\d+$/.test(segment)) {
      key.push([0, Number.parseInt(segment, 10)]);
    } else {
      key.push([1, segment]);
    }
  }
  return key;
}

function selectPreferredReference(jobCoverages) {
  if (jobCoverages.has('coverage-3.11')) {
    return 'coverage-3.11';
  }
  let best = null;
  for (const key of jobCoverages.keys()) {
    if (best === null) {
      best = key;
      continue;
    }
    const candidateKey = naturalSortKey(key);
    const bestKey = naturalSortKey(best);
    const compareLength = Math.max(candidateKey.length, bestKey.length);
    let replaced = false;
    for (let index = 0; index < compareLength; index += 1) {
      const cand = candidateKey[index];
      const existing = bestKey[index];
      if (!cand) {
        break;
      }
      if (!existing) {
        replaced = true;
        break;
      }
      if (cand[0] !== existing[0]) {
        if (cand[0] < existing[0]) {
          replaced = true;
        }
        break;
      }
      if (cand[1] < existing[1]) {
        replaced = true;
        break;
      }
      if (cand[1] > existing[1]) {
        break;
      }
    }
    if (replaced) {
      best = key;
    }
  }
  return best;
}

function loadJsonFile(filePath) {
  if (!filePath) {
    return null;
  }
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const data = JSON.parse(content);
    return data && typeof data === 'object' ? data : null;
  } catch (error) {
    return null;
  }
}

function loadHistoryNdjson(filePath) {
  if (!filePath || !fs.existsSync(filePath)) {
    return [];
  }
  const result = [];
  try {
    const lines = fs.readFileSync(filePath, 'utf8').split(/\r?\n/);
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      try {
        const parsed = JSON.parse(trimmed);
        if (parsed && typeof parsed === 'object') {
          result.push(parsed);
        }
      } catch (error) {
        // ignore malformed lines
      }
    }
  } catch (error) {
    return [];
  }
  return result;
}

function extractNumber(record, key) {
  if (!record || typeof record !== 'object') {
    return null;
  }
  const value = record[key];
  return safeNumber(value);
}

function computeDelta(latest, previous) {
  if (latest === null || previous === null) {
    return null;
  }
  const delta = safeNumber(latest) - safeNumber(previous);
  if (!Number.isFinite(delta)) {
    return null;
  }
  return Math.round(delta * 100) / 100;
}

async function normalizeCoverageArtifacts({
  core,
  rootDir = 'summary_artifacts',
  statsPath = 'coverage-stats.json',
  deltaPath = 'coverage-delta-output.json',
} = {}) {
  const coverageRoot = path.join(rootDir, 'coverage-runtimes');
  const jobCoverages = new Map();
  for (const [label, directory] of discoverPayloads(coverageRoot)) {
    const value = readCoverage(directory);
    if (value !== null) {
      jobCoverages.set(label, Number.parseFloat(value.toFixed(2)));
    }
  }

  const preferredReference = jobCoverages.size ? selectPreferredReference(jobCoverages) : null;

  const sortedJobs = Array.from(jobCoverages.entries()).sort((a, b) => {
    if (preferredReference && a[0] === preferredReference) {
      if (b[0] === preferredReference) {
        return 0;
      }
      return -1;
    }
    if (preferredReference && b[0] === preferredReference) {
      return 1;
    }
    const keyA = naturalSortKey(a[0]);
    const keyB = naturalSortKey(b[0]);
    const length = Math.max(keyA.length, keyB.length);
    for (let index = 0; index < length; index += 1) {
      const partA = keyA[index];
      const partB = keyB[index];
      if (!partA && !partB) {
        return 0;
      }
      if (!partA) {
        return -1;
      }
      if (!partB) {
        return 1;
      }
      if (partA[0] !== partB[0]) {
        return partA[0] - partB[0];
      }
      if (partA[1] < partB[1]) {
        return -1;
      }
      if (partA[1] > partB[1]) {
        return 1;
      }
    }
    return 0;
  });

  const jobRows = [];
  const tableLines = [];
  let diffReference = null;
  let referenceValue = null;
  if (sortedJobs.length) {
    const [refKey, refValue] = sortedJobs[0];
    diffReference = runtimeFrom(refKey);
    referenceValue = refValue;
    tableLines.push(`| Runtime | Coverage | Δ vs ${diffReference} |`);
    tableLines.push('| --- | --- | --- |');
    sortedJobs.forEach(([name, value], index) => {
      const label = runtimeFrom(name);
      let deltaDisplay = '—';
      let deltaValue = null;
      if (index !== 0 && referenceValue !== null) {
        deltaValue = Number.parseFloat((value - referenceValue).toFixed(2));
        deltaDisplay = `${deltaValue >= 0 ? '+' : ''}${deltaValue.toFixed(2)} pp`;
      }
      jobRows.push({
        name,
        label,
        coverage: value,
        delta_vs_reference: deltaValue,
      });
      tableLines.push(`| ${label} | ${value.toFixed(2)}% | ${deltaDisplay} |`);
    });
  }

  const coverageTable = tableLines.length ? tableLines.join('\n') : '';
  const computedAvg = sortedJobs.length
    ? Number.parseFloat(
        (
          sortedJobs.reduce((acc, [, value]) => acc + value, 0) /
          sortedJobs.length
        ).toFixed(2),
      )
    : null;
  const computedWorst = sortedJobs.length
    ? Number.parseFloat(
        Math.min(...sortedJobs.map(([, value]) => value)).toFixed(2),
      )
    : null;

  const recordPath = findFirst(rootDir, 'coverage-trend.json');
  const historyPath = findFirst(rootDir, 'coverage-trend-history.ndjson');
  const deltaFilePath = findFirst(rootDir, 'coverage-delta.json');

  const latestRecord = loadJsonFile(recordPath);
  const historyRecords = loadHistoryNdjson(historyPath);

  let fallbackLatest = latestRecord;
  let latestId = null;
  if (!fallbackLatest && historyRecords.length) {
    fallbackLatest = historyRecords[historyRecords.length - 1];
  }
  if (fallbackLatest) {
    latestId = [fallbackLatest.run_id, fallbackLatest.run_number];
  }

  let previousRecord = null;
  if (historyRecords.length) {
    for (let index = historyRecords.length - 1; index >= 0; index -= 1) {
      const candidate = historyRecords[index];
      const identifier = [candidate.run_id, candidate.run_number];
      if (
        latestId &&
        identifier[0] === latestId[0] &&
        identifier[1] === latestId[1]
      ) {
        continue;
      }
      previousRecord = candidate;
      break;
    }
    if (!previousRecord && historyRecords.length > 1) {
      previousRecord = historyRecords[historyRecords.length - 2];
    }
  }

  const historyAvgLatest = extractNumber(fallbackLatest, 'avg_coverage');
  const historyWorstLatest = extractNumber(fallbackLatest, 'worst_job_coverage');
  const avgPrev = extractNumber(previousRecord, 'avg_coverage');
  const worstPrev = extractNumber(previousRecord, 'worst_job_coverage');

  const avgLatestValue = computedAvg !== null ? computedAvg : historyAvgLatest;
  const worstLatestValue =
    computedWorst !== null ? computedWorst : historyWorstLatest;

  const stats = {
    avg_latest: avgLatestValue ?? null,
    avg_previous: avgPrev ?? null,
    avg_delta: computeDelta(avgLatestValue, avgPrev),
    worst_latest: worstLatestValue ?? null,
    worst_previous: worstPrev ?? null,
    worst_delta: computeDelta(worstLatestValue, worstPrev),
    history_len: historyRecords.length,
  };

  if (jobRows.length) {
    stats.job_coverages = jobRows;
    stats.job_count = jobRows.length;
  }
  if (coverageTable) {
    stats.coverage_table_markdown = coverageTable;
  }
  if (diffReference) {
    stats.diff_reference = diffReference;
  }

  fs.writeFileSync(statsPath, JSON.stringify(stats), 'utf8');
  core.setOutput('stats_json', JSON.stringify(stats));

  const deltaPayload = loadJsonFile(deltaFilePath);
  if (deltaPayload) {
    fs.writeFileSync(deltaPath, JSON.stringify(deltaPayload), 'utf8');
    core.setOutput('delta_json', JSON.stringify(deltaPayload));
  }

  return {
    stats,
    delta: deltaPayload,
    jobCoverages,
  };
}

module.exports = {
  readCoverage,
  discoverPayloads,
  runtimeFrom,
  naturalSortKey,
  normalizeCoverageArtifacts,
};
