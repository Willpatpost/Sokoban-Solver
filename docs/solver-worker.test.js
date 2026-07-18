const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function loadWorker(postMessage = () => {}) {
  const source = fs.readFileSync(path.join(__dirname, "solver-worker.js"), "utf8");
  const context = {
    postMessage,
    onmessage: null,
    console,
  };
  vm.runInNewContext(source, context, {filename: "solver-worker.js"});
  return context;
}

function stateFromRows(rows) {
  let robot = null;
  const boxes = [];
  rows.forEach((row, y) => [...row].forEach((cell, x) => {
    if (cell === "R") robot = [y, x];
    if (cell === "X" || (/[A-Z]/.test(cell) && !"ORS".includes(cell))) {
      boxes.push([`${y},${x}`, cell]);
    }
  }));
  return {rows, robot, boxes};
}

test("browser worker solves a one-push dedicated-box puzzle", () => {
  const worker = loadWorker();
  const result = worker.search({
    algorithm: "push-astar",
    state: stateFromRows(["OOOOO", "O R O", "O A O", "O a O", "OOOOO"]),
  });

  assert.deepEqual(Array.from(result.path), ["Down"]);
  assert.equal(typeof result.visited, "number");
});

test("browser worker prunes static dead-square pushes", () => {
  const worker = loadWorker();
  const board = worker.parse({
    rows: [
      "OOOOOO",
      "O    O",
      "O RX O",
      "O  S O",
      "OOOOOO",
    ],
  });
  const state = {
    robot: [2, 2],
    boxes: [[2, 3, "X"]],
    cost: 0,
  };

  const moves = worker.neighbors(state, board).map(next => next.move);
  assert.equal(worker.staticDead(2, 4, board, "X"), true);
  assert.equal(moves.includes("Right"), false);
});

test("browser worker prunes 2x2 box deadlocks", () => {
  const worker = loadWorker();
  const board = worker.parse({
    rows: [
      "OOOOOO",
      "O    O",
      "O RXXO",
      "O  XOO",
      "O  SSO",
      "OOOOOO",
    ],
  });
  const boxes = [[2, 3, "X"], [2, 4, "X"], [3, 3, "X"]];

  assert.equal(worker.creates2x2Deadlock(boxes, board, [3, 3]), true);
});

test("bidirectional sides emit compatible compact records", () => {
  const rows = ["OOOOO", "O R O", "O A O", "O a O", "OOOOO"];
  const state = stateFromRows(rows);
  const forwardMessages = [], reverseMessages = [];
  const forward = loadWorker(message => forwardMessages.push(message));
  const reverse = loadWorker(message => reverseMessages.push(message));

  forward.bidirectionalSide({mode: "bidir-forward", state});
  reverse.bidirectionalSide({
    mode: "bidir-reverse",
    state,
    reverseShard: {index: 0, count: 1},
  });

  const records = messages => messages
    .filter(message => message.type === "records")
    .flatMap(message => message.records);
  const forwardRecords = records(forwardMessages);
  const reverseIds = new Set(records(reverseMessages).map(record => record.id));
  assert.equal(forwardRecords.some(record => reverseIds.has(record.id)), true);
  assert.equal(forwardRecords.every(record => !("key" in record)), true);
  assert.equal(forwardRecords.every(record => typeof record.segment === "string"), true);
});

test("Hungarian matching enforces distinct goals and detects Hall deadlocks", () => {
  const worker = loadWorker();
  assert.equal(worker.minimumAssignmentCost([
    [0, 10, 10],
    [0, 1, 10],
    [10, 1, 0],
  ]), 1);
  assert.equal(worker.minimumAssignmentCost([
    [0, Infinity],
    [0, Infinity],
  ]), Infinity);
});

test("reverse search charges one unit per pull regardless of walking", () => {
  const worker = loadWorker();
  const board = worker.parse({rows: ["OOOOO", "O   O", "O   O", "O a O", "OOOOO"]});
  const state = {robot: [2, 2], boxes: [[3, 2, "A"]], cost: 0};
  const pulls = worker.reversePullNeighbors(state, board);
  assert.equal(pulls.some(next => next.cost === 1), true);
});

test("frozen components are pruned without rejecting movable box groups", () => {
  const worker = loadWorker();
  const frozenBoard = worker.parse({rows: ["OOOOOOO", "O    SO", "OOOOOOO"]});
  const frozenBoxes = [[1, 2, "X"], [1, 3, "X"], [1, 4, "X"]];
  assert.equal(
    worker.createsFrozenComponentDeadlock(frozenBoxes, frozenBoard, [1, 3]),
    true,
  );

  const openBoard = worker.parse({rows: [
    "OOOOOOO",
    "O     O",
    "O    SO",
    "O     O",
    "OOOOOOO",
  ]});
  assert.equal(
    worker.createsFrozenComponentDeadlock(frozenBoxes.map(([y, x, label]) => [y + 1, x, label]), openBoard, [2, 3]),
    false,
  );
});

test("push beam returns a replayable solution", () => {
  const worker = loadWorker();
  const result = worker.search({
    algorithm: "push-beam",
    beamWidth: 20,
    state: stateFromRows(["OOOOO", "O R O", "O A O", "O a O", "OOOOO"]),
  });
  assert.deepEqual(Array.from(result.path), ["Down"]);
});
