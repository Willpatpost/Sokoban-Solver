const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const DIRECTIONS = [
  ["Up", -1, 0],
  ["Down", 1, 0],
  ["Left", 0, -1],
  ["Right", 0, 1],
];

function loadWorker(messages = []) {
  const source = ["solver-engine.js", "solver-search.js"]
    .map(file => fs.readFileSync(path.join(__dirname, file), "utf8"))
    .join("\n");
  const context = {postMessage: message => messages.push(message), console};
  vm.runInNewContext(source, context, {filename: "exact-kernel.js"});
  return context;
}

function stateFromRows(rows) {
  let robot = null;
  const boxes = [];
  rows.forEach((row, y) => [...row].forEach((cell, x) => {
    if (cell === "R") robot = [y, x];
    if (cell === "X" || (cell >= "A" && cell <= "Z" && !"ORS".includes(cell))) {
      boxes.push([`${y},${x}`, cell]);
    }
  }));
  return {rows, robot, boxes};
}

function referenceProblem(rows) {
  const floor = new Set(), goals = new Map(), boxes = new Map();
  let robot = null;
  rows.forEach((row, y) => [...row].forEach((cell, x) => {
    const key = `${y},${x}`;
    if (cell !== "O") floor.add(key);
    if (cell === "R") robot = [y, x];
    if (cell === "S") goals.set(key, "X");
    if (cell >= "a" && cell <= "z") goals.set(key, cell.toUpperCase());
    if (cell === "X" || (cell >= "A" && cell <= "Z" && !"ORS".includes(cell))) {
      boxes.set(key, cell);
    }
  }));
  return {floor, goals, boxes, robot};
}

function referenceIdentity(robot, boxes) {
  return `${robot.join(",")}|${[...boxes].sort().map(([key, label]) => `${label}@${key}`).join(";")}`;
}

function referenceGoal(problem, boxes) {
  return [...boxes].every(([key, label]) => problem.goals.get(key) === label);
}

function referenceMinPushes(rows) {
  const problem = referenceProblem(rows);
  const start = {robot: problem.robot, boxes: problem.boxes, pushes: 0};
  const deque = [start];
  const distances = new Map([[referenceIdentity(start.robot, start.boxes), 0]]);
  while (deque.length) {
    const current = deque.shift();
    const identity = referenceIdentity(current.robot, current.boxes);
    if (distances.get(identity) !== current.pushes) continue;
    if (referenceGoal(problem, current.boxes)) return current.pushes;
    for (const [_move, dy, dx] of DIRECTIONS) {
      const [y, x] = current.robot;
      const destination = `${y + dy},${x + dx}`;
      if (!problem.floor.has(destination)) continue;
      const boxes = new Map(current.boxes);
      let pushed = 0;
      if (boxes.has(destination)) {
        const beyond = `${y + 2 * dy},${x + 2 * dx}`;
        if (!problem.floor.has(beyond) || boxes.has(beyond)) continue;
        const label = boxes.get(destination);
        boxes.delete(destination);
        boxes.set(beyond, label);
        pushed = 1;
      }
      const next = {robot: [y + dy, x + dx], boxes, pushes: current.pushes + pushed};
      const nextIdentity = referenceIdentity(next.robot, boxes);
      if (next.pushes >= (distances.get(nextIdentity) ?? Infinity)) continue;
      distances.set(nextIdentity, next.pushes);
      if (pushed) deque.push(next);
      else deque.unshift(next);
    }
  }
  return null;
}

function replayPushes(rows, moves) {
  const problem = referenceProblem(rows);
  let robot = problem.robot, boxes = problem.boxes, pushes = 0;
  for (const move of moves || []) {
    const direction = DIRECTIONS.find(([name]) => name === move);
    assert.ok(direction, `unknown move ${move}`);
    const [, dy, dx] = direction;
    const [y, x] = robot;
    const destination = `${y + dy},${x + dx}`;
    assert.ok(problem.floor.has(destination), `illegal move ${move}`);
    if (boxes.has(destination)) {
      const beyond = `${y + 2 * dy},${x + 2 * dx}`;
      assert.ok(problem.floor.has(beyond) && !boxes.has(beyond), `illegal push ${move}`);
      const nextBoxes = new Map(boxes);
      const label = nextBoxes.get(destination);
      nextBoxes.delete(destination);
      nextBoxes.set(beyond, label);
      boxes = nextBoxes;
      pushes++;
    }
    robot = [y + dy, x + dx];
  }
  assert.equal(referenceGoal(problem, boxes), true);
  return pushes;
}

function tinyPlacements() {
  const cells = [];
  for (let y = 1; y <= 2; y++) for (let x = 1; x <= 3; x++) cells.push([y, x]);
  const cases = [];
  for (let box = 0; box < cells.length; box++) {
    for (let goal = 0; goal < cells.length; goal++) {
      if (goal === box) continue;
      for (let robot = 0; robot < cells.length; robot++) {
        if (robot === box || robot === goal) continue;
        const grid = Array.from({length: 4}, () => [..."OOOOO"]);
        for (const [y, x] of cells) grid[y][x] = " ";
        grid[cells[box][0]][cells[box][1]] = "X";
        grid[cells[goal][0]][cells[goal][1]] = "S";
        grid[cells[robot][0]][cells[robot][1]] = "R";
        cases.push(grid.map(row => row.join("")));
      }
    }
  }
  return cases;
}

const AUTHORED = [
  ["OOOOOOO", "O R   O", "O XX  O", "O SS  O", "OOOOOOO"],
  ["OOOOOOO", "O R   O", "O AB  O", "O ab  O", "OOOOOOO"],
  ["OOOOOOO", "O S S O", "O X X O", "O  R  O", "OOOOOOO"],
  ["OOOOOO", "OX R O", "O   SO", "OOOOOO"],
];

test("exact push A* matches an independent exhaustive step-state oracle", () => {
  const worker = loadWorker();
  for (const rows of [...tinyPlacements(), ...AUTHORED]) {
    const expected = referenceMinPushes(rows);
    const result = worker.search({
      algorithm: "push-astar",
      state: stateFromRows(rows),
    });
    if (expected === null) {
      assert.equal(result.status, "proven-unsolvable");
      assert.equal(result.path, null);
    } else {
      assert.equal(result.status, "solved");
      assert.equal(replayPushes(rows, result.path), expected);
    }
  }
});

test("exact IDA options preserve oracle solvability and optimal pushes", () => {
  for (const rows of AUTHORED) {
    const expected = referenceMinPushes(rows);
    for (const options of [
      {lockProvenCommitments: false, forcedMacros: false, transpositionLimit: 1},
      {lockProvenCommitments: true, forcedMacros: false, transpositionLimit: 100},
      {lockProvenCommitments: false, forcedMacros: true, transpositionLimit: 4},
    ]) {
      const worker = loadWorker();
      const result = worker.search({
        algorithm: "push-ida-star",
        state: stateFromRows(rows),
        upperBound: Infinity,
        maxVisited: 1000000,
        ...options,
      });
      if (expected === null) assert.equal(result.status, "proven-unsolvable");
      else {
        assert.equal(result.status, "solved");
        assert.equal(replayPushes(rows, result.path), expected);
      }
    }
  }
});

test("finite invalid incumbents cannot become false proofs", () => {
  const rows = AUTHORED[0];
  const expected = referenceMinPushes(rows);
  const worker = loadWorker();
  const bounded = worker.search({
    algorithm: "push-ida-star",
    state: stateFromRows(rows),
    upperBound: expected - 1,
    maxVisited: 100000,
  });
  assert.equal(bounded.status, "cutoff");
  assert.notEqual(bounded.status, "proven-unsolvable");
  const unbounded = worker.search({
    algorithm: "push-ida-star",
    state: stateFromRows(rows),
    upperBound: Infinity,
    maxVisited: 100000,
  });
  assert.equal(unbounded.status, "solved");
  assert.equal(replayPushes(rows, unbounded.path), expected);

  const beyondDefault = [
    "O".repeat(306),
    `ORX${" ".repeat(300)}SO`,
    "O".repeat(306),
  ];
  const defaultBound = worker.search({
    algorithm: "push-ida-star",
    state: stateFromRows(beyondDefault),
  });
  assert.equal(defaultBound.status, "cutoff");
  assert.notEqual(defaultBound.status, "proven-unsolvable");
});

test("persistent shard union preserves exact outcomes", () => {
  for (const rows of AUTHORED) {
    const expected = referenceMinPushes(rows);
    const results = [0, 1, 2].map(index => loadWorker().search({
      algorithm: "push-ida-star",
      state: stateFromRows(rows),
      upperBound: Infinity,
      maxVisited: 1000000,
      exactShard: {index, count: 3, depth: 2},
      forcedMacros: false,
    }));
    if (expected === null) {
      assert.ok(results.every(result => result.status === "proven-unsolvable"));
    } else {
      const solved = results.filter(result => result.status === "solved");
      assert.ok(solved.length >= 1);
      assert.equal(
        Math.min(...solved.map(result => replayPushes(rows, result.path))),
        expected,
      );
    }
  }
});

test("exact checkpoints resume identically after many interruption points", () => {
  const rows = AUTHORED[0];
  const payload = {
    algorithm: "push-ida-star",
    state: stateFromRows(rows),
    upperBound: Infinity,
    maxVisited: 1000000,
    solverBuild: "test-build",
    forcedMacros: false,
  };
  const uninterrupted = loadWorker().search(payload);
  const expectedPushes = replayPushes(rows, uninterrupted.path);
  for (const slice of [1, 2, 3, 5, 8]) {
    let checkpoint = null, result = null;
    for (let resumes = 0; resumes < 10000; resumes++) {
      result = loadWorker().search({
        ...payload,
        pauseAfterVisited: slice,
        resumeExactCheckpoint: checkpoint,
      });
      if (result.status !== "cutoff") break;
      assert.equal(result.terminationReason, "checkpoint-yield");
      checkpoint = JSON.parse(JSON.stringify(result.exactCheckpoint));
    }
    assert.equal(result.status, uninterrupted.status);
    assert.equal(replayPushes(rows, result.path), expectedPushes);
  }
});

test("exact checkpoints reject a different board or build", () => {
  const rows = AUTHORED[0];
  const base = {
    algorithm: "push-ida-star",
    state: stateFromRows(rows),
    upperBound: Infinity,
    maxVisited: 1000000,
    solverBuild: "build-a",
    pauseAfterVisited: 1,
  };
  const paused = loadWorker().search(base);
  assert.equal(paused.terminationReason, "checkpoint-yield");
  const wrongBuild = loadWorker().search({
    ...base,
    pauseAfterVisited: undefined,
    solverBuild: "build-b",
    resumeExactCheckpoint: paused.exactCheckpoint,
  });
  assert.equal(wrongBuild.status, "failed");
  assert.equal(wrongBuild.terminationReason, "checkpoint-incompatible");
  const wrongBoard = loadWorker().search({
    ...base,
    pauseAfterVisited: undefined,
    state: stateFromRows(AUTHORED[1]),
    resumeExactCheckpoint: paused.exactCheckpoint,
  });
  assert.equal(wrongBoard.status, "failed");
});

test("explicit exact traversal stack handles deep push paths", () => {
  const pushes = 600;
  const rows = [
    "O".repeat(pushes + 5),
    `ORX${" ".repeat(pushes - 1)}SO`,
    "O".repeat(pushes + 5),
  ];
  const result = loadWorker().search({
    algorithm: "push-ida-star",
    state: stateFromRows(rows),
    upperBound: Infinity,
    maxVisited: 1000000,
    forcedMacros: false,
  });
  assert.equal(result.status, "solved");
  assert.equal(replayPushes(rows, result.path), pushes);
  assert.ok(result.maxDepth >= pushes);
});
