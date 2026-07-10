const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function loadWorker() {
  const source = fs.readFileSync(path.join(__dirname, "solver-worker.js"), "utf8");
  const context = {
    postMessage() {},
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
