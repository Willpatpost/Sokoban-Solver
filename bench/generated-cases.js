"use strict";

const {LEVELS} = require("../docs/levels.js");

function mirrorRows(rows) {
  return rows.map(row => [...row].reverse().join(""));
}

function rotateRows(rows) {
  return [...rows].reverse().map(row => [...row].reverse().join(""));
}

function permuteLabels(rows, mapping) {
  return rows.map(row => [...row].map(cell => {
    const upper = cell.toUpperCase();
    if (!mapping[upper]) return cell;
    return cell === upper ? mapping[upper] : mapping[upper].toLowerCase();
  }).join(""));
}

const GENERATED_CASES = [
  {
    name: "generated mirrored tiny",
    rows: mirrorRows(LEVELS.tiny),
    algorithm: "portfolio",
    timeoutMs: 10000,
    weight: 2,
    payload: {maxVisited: 60000},
  },
  {
    name: "generated rotated medium",
    rows: rotateRows(LEVELS.medium),
    algorithm: "push-beam",
    timeoutMs: 20000,
    weight: 4,
    payload: {maxVisited: 180000, beamWidth: 360, maxDepth: 180, sequenceMacros: true},
  },
  {
    name: "generated relabeled medium",
    rows: permuteLabels(LEVELS.medium, {A: "E", B: "F", C: "G", D: "H"}),
    algorithm: "push-beam",
    timeoutMs: 20000,
    weight: 4,
    payload: {maxVisited: 180000, beamWidth: 360, maxDepth: 180, sequenceMacros: true},
  },
  {
    name: "generated premature-goal ordering",
    rows: [
      "OOOOOOO",
      "O SS  O",
      "O     O",
      "O XXR O",
      "O     O",
      "OOOOOOO",
    ],
    algorithm: "push-astar",
    timeoutMs: 10000,
    weight: 2,
    payload: {maxVisited: 60000},
  },
];

module.exports = {GENERATED_CASES, mirrorRows, permuteLabels, rotateRows};
