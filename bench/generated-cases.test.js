const assert = require("node:assert/strict");
const test = require("node:test");

const {LEVELS} = require("../docs/levels.js");
const {
  GENERATED_CASES,
  STRATEGIC_CASES,
  mirrorRows,
  permuteLabels,
  rotateRows,
} = require("./generated-cases.js");

test("generated transforms are deterministic and preserve board dimensions", () => {
  const tinyMirror = mirrorRows(LEVELS.tiny);
  assert.deepEqual(mirrorRows(tinyMirror), LEVELS.tiny);
  assert.deepEqual(rotateRows(rotateRows(LEVELS.medium)), LEVELS.medium);
  assert.deepEqual(tinyMirror.map(row => row.length), LEVELS.tiny.map(row => row.length));
});

test("label permutations change boxes and their matching goals together", () => {
  const rows = permuteLabels(["OAaBbXO"], {A: "E", B: "F"});
  assert.deepEqual(rows, ["OEeFfXO"]);
});

test("generated benchmark cases have unique names and explicit state budgets", () => {
  assert.equal(new Set(GENERATED_CASES.map(caseSpec => caseSpec.name)).size, GENERATED_CASES.length);
  for (const caseSpec of GENERATED_CASES) {
    assert.ok(Array.isArray(caseSpec.rows));
    assert.ok(caseSpec.payload.maxVisited > 0);
    assert.ok(caseSpec.timeoutMs > 0);
  }
});

test("strategic benchmark seeds expand into deterministic transformation families", () => {
  for (const family of ["typed doorway import", "exact room packing", "corral reopening"]) {
    assert.ok(STRATEGIC_CASES.filter(caseSpec => caseSpec.name.includes(family)).length >= 2);
  }
  const relabeled = STRATEGIC_CASES.find(caseSpec => caseSpec.name.endsWith("relabeled"));
  assert.ok(relabeled);
  assert.ok(relabeled.rows.some(row => row.includes("E")));
  assert.ok(relabeled.rows.some(row => row.includes("e")));
});
