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

function laneWarehouseRows(boxCount, {typed = false, pushDistance = 2} = {}) {
  if (!Number.isInteger(boxCount) || boxCount < 2 || boxCount > 20) {
    throw new RangeError("boxCount must be an integer between 2 and 20.");
  }
  if (!Number.isInteger(pushDistance) || pushDistance < 1 || pushDistance > 20) {
    throw new RangeError("pushDistance must be an integer between 1 and 20.");
  }
  const boxColumn = 4;
  const goalColumn = boxColumn + pushDistance;
  const width = goalColumn + 3;
  const wall = "O".repeat(width);
  const empty = `O${" ".repeat(width - 2)}O`;
  const rows = [wall, empty];
  for (let index = 0; index < boxCount; index++) {
    const row = [...empty];
    if (index === 0) row[2] = "R";
    const label = typed ? String.fromCharCode(65 + index) : "X";
    row[boxColumn] = label;
    row[goalColumn] = typed ? label.toLowerCase() : "S";
    rows.push(row.join(""));
  }
  rows.push(empty, wall);
  return rows;
}

function certifiedMultiboxCase(name, boxCount, options = {}) {
  return {
    name: `generated certified multibox ${name}`,
    rows: laneWarehouseRows(boxCount, options),
    algorithm: "push-astar",
    timeoutMs: 15000,
    weight: boxCount,
    payload: {maxVisited: 200000},
    family: "certified-multibox",
    certification: "exact-reference",
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

const CERTIFIED_MULTIBOX_CASES = [
  certifiedMultiboxCase("three lane", 3),
  certifiedMultiboxCase("typed three lane", 3, {typed: true}),
  certifiedMultiboxCase("four lane long push", 4, {pushDistance: 3}),
];

function seededFamilyCase(family, seed, profile) {
  const random = (() => {
    let value = seed >>> 0;
    return () => {
      value ^= value << 13; value ^= value >>> 17; value ^= value << 5;
      return (value >>> 0) / 0x100000000;
    };
  })();
  const transforms = [profile.rows, mirrorRows(profile.rows), rotateRows(profile.rows)];
  let rows = transforms[Math.floor(random() * transforms.length)];
  if (profile.labels && random() >= .5) rows = permuteLabels(rows, profile.labels);
  const boxCount = rows.join("").match(/X|[A-NP-QT-Z]/g)?.length || 0;
  return {
    name: `generated ${family} family`,
    rows,
    algorithm: "push-astar",
    timeoutMs: 15000,
    weight: boxCount,
    payload: {maxVisited: 200000},
    family,
    certification: "independent-exact-push",
    reviewedExpectation: {solved: true, pushes: profile.pushes},
  };
}

// Seeds stay in the harness and are never passed in the solver payload.
const CERTIFIED_STRATEGIC_FAMILIES = Object.freeze([
  seededFamilyCase("bottleneck", 0xb0771e, {
    rows: ["OOOOOOO", "O  S  O", "OOO OOO", "O  X  O", "O  R  O", "O     O", "OOOOOOO"],
    pushes: 2,
  }),
  seededFamilyCase("staging-capacity", 0x57a613, {
    rows: ["OOOOOOOO", "O SS   O", "OO  OOOO", "O XX   O", "O  R   O", "O      O",
      "OOOOOOOO"],
    pushes: 4,
  }),
  seededFamilyCase("coupled-room-ordering", 0xc0a913, {
    rows: ["OOOOOOOOO", "O a   b O", "O       O", "OOOO OOOO", "O  A B  O",
      "O   R   O", "O       O", "OOOOOOOOO"],
    pushes: 12,
    labels: {A: "E", B: "F"},
  }),
  seededFamilyCase("dependency-cycle", 0xde9e0d, {
    rows: ["OOOOOOOOO", "O S S S O", "O       O", "O X O X O", "O   X   O",
      "O   R   O", "O       O", "OOOOOOOOO"],
    pushes: 9,
  }),
  seededFamilyCase("multi-gate", 0x6a7e55, {
    rows: ["OOOOOOOOOOO", "O S S S   O", "OO O O OOOO", "O X X X   O",
      "O    R    O", "O         O", "OOOOOOOOOOO"],
    pushes: 6,
  }),
]);

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
  ...CERTIFIED_MULTIBOX_CASES,
  ...CERTIFIED_STRATEGIC_FAMILIES,
];

module.exports = {
  CERTIFIED_MULTIBOX_CASES,
  CERTIFIED_STRATEGIC_FAMILIES,
  GENERATED_CASES,
  STRATEGIC_CASES,
  laneWarehouseRows,
  mirrorRows,
  permuteLabels,
  rotateRows,
  strategicVariants,
};
