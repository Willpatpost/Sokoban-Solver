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

function guidedCase(name, rows) {
  return {
    name,
    rows,
    algorithm: "push-beam",
    timeoutMs: 10000,
    weight: 2,
    payload: {maxVisited: 60000, beamWidth: 120, maxDepth: 40},
  };
}

function strategicVariants(name, rows, labelMapping = null) {
  const candidates = [
    ["base", rows],
    ["mirrored", mirrorRows(rows)],
    ["rotated", rotateRows(rows)],
  ];
  if (labelMapping) candidates.push(["relabeled", permuteLabels(rows, labelMapping)]);
  const seen = new Set();
  return candidates
    .filter(([, candidateRows]) => {
      const signature = candidateRows.join("\n");
      if (seen.has(signature)) return false;
      seen.add(signature);
      return true;
    })
    .map(([variant, candidateRows]) => guidedCase(`generated ${name} ${variant}`, candidateRows));
}

const STRATEGIC_CASES = [
  ...strategicVariants("typed doorway import", [
    "OOOOOOOOO",
    "O R A a O",
    "O       O",
    "O       O",
    "OOOOAOOOO",
    "O   a   O",
    "O       O",
    "OOOOOOOOO",
  ], {A: "E"}),
  ...strategicVariants("exact room packing", [
    "OOOOOOO",
    "O R   O",
    "OOO OOO",
    "O S S O",
    "O X X O",
    "O     O",
    "OOOOOOO",
  ]),
  ...strategicVariants("corral reopening", ["OOOOOOO", "OR X SO", "OOOOOOO"]),
];

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
  ...STRATEGIC_CASES,
];

module.exports = {
  GENERATED_CASES,
  STRATEGIC_CASES,
  mirrorRows,
  permuteLabels,
  rotateRows,
  strategicVariants,
};
