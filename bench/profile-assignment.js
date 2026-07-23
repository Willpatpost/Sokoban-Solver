"use strict";

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {performance} = require("node:perf_hooks");

const root = path.resolve(__dirname, "..");
const source = fs.readFileSync(path.join(root, "docs", "solver-engine.js"), "utf8");
const context = {console, performance};
vm.runInNewContext(source, context, {filename: "solver-engine.js"});

function matrix(size, offset) {
  return Array.from({length: size}, (_, row) =>
    Array.from({length: size}, (_, column) =>
      ((row + 3) * (column + 5) + offset * 7) % 31));
}

function median(values) {
  return [...values].sort((left, right) => left - right)[Math.floor(values.length / 2)];
}

function time(operation, iterations) {
  const started = performance.now();
  for (let index = 0; index < iterations; index++) operation(index);
  return performance.now() - started;
}

const results = [];
for (let size = 2; size <= 10; size++) {
  const base = matrix(size, 0);
  const previous = context.minimumAssignment(base);
  const iterations = Math.max(500, Math.floor(12000 / size));
  const fullSamples = [], repairSamples = [];
  for (let repeat = 0; repeat < 7; repeat++) {
    fullSamples.push(time(index => {
      const changed = base.map(row => [...row]);
      changed[index % size] = matrix(size, index + repeat)[index % size];
      context.minimumAssignment(changed);
    }, iterations));
    repairSamples.push(time(index => {
      const changed = base.map(row => [...row]);
      const changedRow = index % size;
      changed[changedRow] = matrix(size, index + repeat)[changedRow];
      context.repairMinimumAssignment(previous, changed, changedRow);
    }, iterations));
  }
  const fullMs = median(fullSamples);
  const repairMs = median(repairSamples);
  results.push({
    size,
    iterations,
    fullMs: Math.round(fullMs * 1000) / 1000,
    repairMs: Math.round(repairMs * 1000) / 1000,
    ratio: Math.round((repairMs / fullMs) * 1000) / 1000,
  });
}
process.stdout.write(`${JSON.stringify({runtime: "javascript", results}, null, 2)}\n`);
