"use strict";

const fs = require("node:fs");
const baselines = require("./huge-performance-baselines.json");

function evaluateHugeResults(results, baseline = baselines) {
  const failures = [];
  if (results.length !== baseline.expectedCases) {
    failures.push(`cases: expected ${baseline.expectedCases}, received ${results.length}`);
  }
  const solved = results.filter(result => result.solved).length;
  if (solved !== baseline.expectedSolved) {
    failures.push(`solved: expected ${baseline.expectedSolved}, received ${solved}`);
  }
  for (const result of results) {
    const label = result.name || result.level || "unnamed case";
    if (!result.valid) failures.push(`${label}: returned path failed replay validation`);
    if (result.error) failures.push(`${label}: ${result.error}`);
    if (result.timeout) failures.push(`${label}: timed out`);
    for (const [field, maximum] of Object.entries(baseline.maximumPerCase)) {
      const actual = result[field];
      if (!Number.isFinite(actual)) {
        failures.push(`${label}: ${field} telemetry is missing`);
        continue;
      }
      if (actual > maximum) {
        failures.push(`${label}: ${field} maximum ${maximum}, received ${actual}`);
      }
    }
    for (const [field, maximum] of Object.entries(
      baseline.maximumPerformancePerCase || {},
    )) {
      const actual = result.performance?.[field];
      if (!Number.isFinite(actual)) {
        failures.push(`${label}: performance.${field} telemetry is missing`);
      } else if (actual > maximum) {
        failures.push(
          `${label}: performance.${field} maximum ${maximum}, received ${actual}`,
        );
      }
    }
    for (const [field, maximum] of Object.entries(
      baseline.maximumMemoryPerCase || {},
    )) {
      const actual = result.performance?.memory?.[field];
      if (!Number.isFinite(actual)) {
        failures.push(`${label}: performance.memory.${field} telemetry is missing`);
      } else if (actual > maximum) {
        failures.push(
          `${label}: performance.memory.${field} maximum ${maximum}, received ${actual}`,
        );
      }
    }
  }
  return failures;
}

function readCaseResults(filePath) {
  const contents = fs.readFileSync(filePath);
  const decoded = contents[0] === 0xff && contents[1] === 0xfe
    ? contents.toString("utf16le")
    : contents.toString("utf8");
  return decoded.replace(/^\uFEFF/, "")
    .split(/\r?\n/)
    .map(line => line.trimStart())
    .filter(line => line.startsWith('{"type":"case"'))
    .map(line => {
      const {type: _type, ...result} = JSON.parse(line);
      return result;
    });
}

function main() {
  const filePath = process.argv[2];
  if (!filePath) throw new Error("Usage: node bench/huge-performance-gate.js RESULTS.jsonl");
  const results = readCaseResults(filePath);
  const failures = evaluateHugeResults(results);
  process.stdout.write(`${JSON.stringify({
    schemaVersion: baselines.schemaVersion,
    reviewedBuild: baselines.reviewedBuild,
    cases: results.length,
    failures,
  }, null, 2)}\n`);
  if (failures.length) process.exitCode = 1;
}

if (require.main === module) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`${error.stack || error.message}\n`);
    process.exitCode = 1;
  }
}

module.exports = {evaluateHugeResults, readCaseResults};
