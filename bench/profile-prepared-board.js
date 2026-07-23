"use strict";

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {performance} = require("node:perf_hooks");
const {LEVELS, stateFromRows} = require("../docs/levels.js");

function loadEngine() {
  const source = ["solver-engine.js", "solver-search.js"]
    .map(file => fs.readFileSync(path.join(__dirname, "..", "docs", file), "utf8"))
    .join("\n");
  const context = {console, postMessage() {}};
  vm.runInNewContext(source, context, {filename: "solver-engine.js"});
  return context;
}

function median(values) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.floor(ordered.length / 2)];
}

const worker = loadEngine();
const puzzle = stateFromRows(LEVELS.huge);
const rounds = 7;
const buildMs = [];
const cloneMs = [];
const hydrateMs = [];
let lastAnalysis;

for (let round = 0; round < rounds; round++) {
  let started = performance.now();
  lastAnalysis = worker.analyzePuzzleForSearch(puzzle);
  buildMs.push(performance.now() - started);

  started = performance.now();
  const cloned = structuredClone(lastAnalysis.preparedBoard);
  cloneMs.push(performance.now() - started);

  started = performance.now();
  worker.parse(puzzle, cloned);
  hydrateMs.push(performance.now() - started);
}

console.log(JSON.stringify({
  schemaVersion: 1,
  build: "2026-07-23.28",
  runtime: process.version,
  level: "huge",
  rounds,
  preparedBytesEstimate: lastAnalysis.preparedBoardStats.estimatedBytes,
  goalTables: lastAnalysis.preparedBoardStats.goalTables,
  playerDistanceTables: lastAnalysis.preparedBoardStats.playerDistanceTables,
  graphNodes: lastAnalysis.preparedBoardStats.graphNodes,
  graphEdges: lastAnalysis.preparedBoardStats.graphEdges,
  medianBuildMs: Number(median(buildMs).toFixed(3)),
  medianStructuredCloneMs: Number(median(cloneMs).toFixed(3)),
  medianHydrateMs: Number(median(hydrateMs).toFixed(3)),
}, null, 2));
