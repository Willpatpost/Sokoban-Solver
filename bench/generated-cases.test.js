const assert = require("node:assert/strict");
const test = require("node:test");

const {LEVELS} = require("../docs/levels.js");
const {
  CERTIFIED_MULTIBOX_CASES,
  CERTIFIED_STRATEGIC_FAMILIES,
  GENERATED_CASES,
  STRATEGIC_CASES,
  laneWarehouseRows,
  mirrorRows,
  permuteLabels,
  rotateRows,
} = require("./generated-cases.js");
const {parseRows} = require("./case-runner.js");

const DIRECTIONS = [[0, 1], [-1, 0], [1, 0], [0, -1]];

function position(y, x) {
  return `${y},${x}`;
}

function exactPushSolve(rows, maxStates = 200000) {
  const initial = parseRows(rows);
  const goalsByLabel = new Map();
  for (const [cell, label] of initial.board.goals) {
    if (!goalsByLabel.has(label)) goalsByLabel.set(label, []);
    goalsByLabel.get(label).push(cell.split(",").map(Number));
  }
  const matchingCost = boxes => {
    const boxesByLabel = new Map();
    for (const [cell, label] of boxes) {
      if (!boxesByLabel.has(label)) boxesByLabel.set(label, []);
      boxesByLabel.get(label).push(cell.split(",").map(Number));
    }
    let total = 0;
    for (const [label, positions] of boxesByLabel) {
      const goals = goalsByLabel.get(label);
      const assign = (index, remaining) => {
        if (index === positions.length) return 0;
        let best = Infinity;
        for (let goalIndex = 0; goalIndex < remaining.length; goalIndex++) {
          const [y, x] = positions[index];
          const [goalY, goalX] = remaining[goalIndex];
          best = Math.min(best, Math.abs(y - goalY) + Math.abs(x - goalX) +
            assign(index + 1, remaining.filter((_, candidate) => candidate !== goalIndex)));
        }
        return best;
      };
      total += assign(0, goals);
    }
    return total;
  };
  const frontier = [{robot: initial.robot, boxes: initial.boxes, pushes: 0,
    priority: matchingCost(initial.boxes)}];
  const bestCost = new Map();
  while (frontier.length && bestCost.size < maxStates) {
    let bestIndex = 0;
    for (let index = 1; index < frontier.length; index++) {
      if (frontier[index].priority < frontier[bestIndex].priority) bestIndex = index;
    }
    const state = frontier.splice(bestIndex, 1)[0];
    const occupied = new Set(state.boxes.keys());
    const reachable = new Set([position(...state.robot)]);
    const walkQueue = [state.robot];
    for (let walkHead = 0; walkHead < walkQueue.length; walkHead++) {
      const [y, x] = walkQueue[walkHead];
      for (const [dy, dx] of DIRECTIONS) {
        const next = position(y + dy, x + dx);
        if (!initial.board.floor.has(next) || occupied.has(next) || reachable.has(next)) continue;
        reachable.add(next);
        walkQueue.push([y + dy, x + dx]);
      }
    }
    const key = `${[...state.boxes].sort().map(entry => entry.join(":"))}|${[...reachable].sort()[0]}`;
    if ((bestCost.get(key) ?? Infinity) <= state.pushes) continue;
    bestCost.set(key, state.pushes);
    if ([...state.boxes].every(([cell, label]) => initial.board.goals.get(cell) === label)) {
      return {solved: true, states: bestCost.size, pushes: state.pushes};
    }
    for (const [cell, label] of state.boxes) {
      const [y, x] = cell.split(",").map(Number);
      for (const [dy, dx] of DIRECTIONS) {
        const support = position(y - dy, x - dx);
        const destination = position(y + dy, x + dx);
        if (!reachable.has(support) || !initial.board.floor.has(destination) || occupied.has(destination)) continue;
        const boxes = new Map(state.boxes);
        boxes.delete(cell);
        boxes.set(destination, label);
        const pushes = state.pushes + 1;
        frontier.push({robot: [y, x], boxes, pushes,
          priority: pushes + matchingCost(boxes)});
      }
    }
  }
  return {solved: false, states: bestCost.size};
}

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

test("procedural lane warehouses are deterministic and reject invalid sizes", () => {
  assert.deepEqual(laneWarehouseRows(3, {typed: true}), laneWarehouseRows(3, {typed: true}));
  assert.throws(() => laneWarehouseRows(1), /boxCount/);
  assert.throws(() => laneWarehouseRows(21), /boxCount/);
  assert.throws(() => laneWarehouseRows(3, {pushDistance: 0}), /pushDistance/);
});

test("larger generated multibox cases are independently certified by exact push search", () => {
  assert.ok(CERTIFIED_MULTIBOX_CASES.some(caseSpec => caseSpec.rows.join("").includes("A")));
  assert.ok(CERTIFIED_MULTIBOX_CASES.some(caseSpec =>
    caseSpec.rows.join("").split("X").length - 1 >= 4));
  for (const caseSpec of CERTIFIED_MULTIBOX_CASES) {
    assert.equal(caseSpec.family, "certified-multibox");
    assert.equal(caseSpec.certification, "exact-reference");
    const result = exactPushSolve(caseSpec.rows);
    assert.equal(result.solved, true, `${caseSpec.name} exhausted ${result.states} states`);
  }
});

test("separately seeded strategic families retain reviewed exact expectations", () => {
  assert.deepEqual(
    CERTIFIED_STRATEGIC_FAMILIES.map(caseSpec => caseSpec.family).sort(),
    ["bottleneck", "coupled-room-ordering", "dependency-cycle", "multi-gate",
      "staging-capacity"],
  );
  for (const caseSpec of CERTIFIED_STRATEGIC_FAMILIES) {
    assert.equal(caseSpec.certification, "independent-exact-push");
    assert.equal(Object.hasOwn(caseSpec.payload, "seed"), false);
    const result = exactPushSolve(caseSpec.rows);
    assert.equal(result.solved, caseSpec.reviewedExpectation.solved, caseSpec.name);
    assert.equal(result.pushes, caseSpec.reviewedExpectation.pushes, caseSpec.name);
  }
  const byFamily = Object.fromEntries(
    CERTIFIED_STRATEGIC_FAMILIES.map(caseSpec => [caseSpec.family, caseSpec.rows.join("\n")]),
  );
  assert.match(byFamily.bottleneck, /OOO OOO/);
  assert.match(byFamily["staging-capacity"], /OO  OOOO|OOOO  OO/);
  assert.match(byFamily["coupled-room-ordering"], /[A-F]/);
  assert.match(byFamily["dependency-cycle"], /O X O X O/);
  assert.match(byFamily["multi-gate"], /OO O O OOOO|OOOO O O OO/);
});
