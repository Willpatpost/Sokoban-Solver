const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const fixtures = require("../shared/sokomind-conformance.json");
const {LEVELS, EMBEDDED_LEVELS, stateFromRows} = require("./levels.js");

function loadWorker() {
  const source = fs.readFileSync(path.join(__dirname, "solver-worker.js"), "utf8");
  const context = {postMessage() {}, onmessage: null, console};
  vm.runInNewContext(source, context, {filename: "solver-worker.js"});
  return context;
}

function errorKind(error) {
  if (error.message.includes("Unsupported symbol")) return "symbol";
  if (error.message.includes("exactly one robot")) return "robot-count";
  return "box-goal-count";
}

test("browser and benchmark level catalogs match the shared canonical catalog", () => {
  assert.deepEqual(LEVELS, fixtures.levels);
  assert.deepEqual(EMBEDDED_LEVELS, fixtures.levels);
});

test("browser worker passes shared valid parsing and rule cases", () => {
  const worker = loadWorker();
  for (const fixture of fixtures.validCases) {
    const {expected, rows} = fixture;
    assert.equal(worker.validatePuzzleRows(rows), true, fixture.id);
    const serialized = stateFromRows(rows);
    const board = worker.parse(serialized);
    const state = {
      robot: serialized.robot,
      boxes: serialized.boxes.map(([position, label]) => [
        ...position.split(",").map(Number), label,
      ]),
    };
    const boxes = serialized.boxes.map(item => Array.from(item)).sort();
    const goals = [...board.goals].map(item => Array.from(item)).sort();
    const legalMoves = Array.from(worker.neighbors(state, board), next => next.move).sort();
    const mechanicalMoves = Array.from(
      worker.neighbors(state, board, false), next => next.move,
    ).sort();

    assert.equal(Math.max(...rows.map(row => row.length)), expected.width, fixture.id);
    assert.equal(rows.length, expected.height, fixture.id);
    assert.equal(board.floor.size, expected.floorCount, fixture.id);
    assert.equal(serialized.robot.join(","), expected.robot, fixture.id);
    assert.deepEqual(boxes, expected.boxes, fixture.id);
    assert.deepEqual(goals, expected.goals, fixture.id);
    assert.deepEqual(legalMoves, expected.legalMoves, fixture.id);
    assert.deepEqual(mechanicalMoves, expected.mechanicalMoves, fixture.id);
    assert.equal(worker.goal(state.boxes, board.goals), expected.solved, fixture.id);
    if (expected.missingWall) {
      assert.equal(board.floor.has(expected.missingWall), false, fixture.id);
    }
  }
});

test("browser worker rejects every shared invalid puzzle definition", () => {
  const worker = loadWorker();
  for (const fixture of fixtures.invalidCases) {
    assert.throws(
      () => worker.validatePuzzleRows(fixture.rows),
      error => errorKind(error) === fixture.errorKind,
      fixture.id,
    );
  }
});
