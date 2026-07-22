const assert = require("node:assert/strict");
const test = require("node:test");

const fixtures = require("../shared/sokomind-conformance.json");
const {moveState, parseRows, validateRows} = require("./case-runner.js");

function errorKind(error) {
  if (error.message.includes("Unsupported symbol")) return "symbol";
  if (error.message.includes("exactly one robot")) return "robot-count";
  return "box-goal-count";
}

test("benchmark replay parser passes shared valid rule cases", () => {
  for (const fixture of fixtures.validCases) {
    const {expected, rows} = fixture;
    const state = parseRows(rows);
    const boxes = [...state.boxes].map(item => Array.from(item)).sort();
    const goals = [...state.board.goals].map(item => Array.from(item)).sort();
    const mechanicalMoves = ["Up", "Down", "Left", "Right"]
      .filter(move => moveState(state, move) !== null)
      .sort();

    assert.equal(state.board.floor.size, expected.floorCount, fixture.id);
    assert.equal(state.robot.join(","), expected.robot, fixture.id);
    assert.deepEqual(boxes, expected.boxes, fixture.id);
    assert.deepEqual(goals, expected.goals, fixture.id);
    assert.deepEqual(mechanicalMoves, expected.mechanicalMoves, fixture.id);
    if (expected.missingWall) {
      assert.equal(state.board.floor.has(expected.missingWall), false, fixture.id);
    }
  }
});

test("benchmark replay parser rejects every shared invalid definition", () => {
  for (const fixture of fixtures.invalidCases) {
    assert.throws(
      () => validateRows(fixture.rows),
      error => errorKind(error) === fixture.errorKind,
      fixture.id,
    );
  }
});
