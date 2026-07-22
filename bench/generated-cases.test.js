const assert = require("node:assert/strict");
const test = require("node:test");

const {LEVELS} = require("../docs/levels.js");
const {GENERATED_CASES, mirrorRows, permuteLabels, rotateRows} = require("./generated-cases.js");

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
