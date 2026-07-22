const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const DIRS = {Up: [-1, 0], Down: [1, 0], Left: [0, -1], Right: [0, 1]};

function loadWorker() {
  const source = fs.readFileSync(path.join(__dirname, "solver-worker.js"), "utf8");
  const context = {postMessage() {}, onmessage: null, console};
  vm.runInNewContext(source, context, {filename: "solver-worker.js"});
  return context;
}

function stateFromRows(rows) {
  let robot = null;
  const boxes = [];
  rows.forEach((row, y) => [...row].forEach((cell, x) => {
    if (cell === "R") robot = [y, x];
    if (cell === "X" || (/[A-Z]/.test(cell) && !"ORS".includes(cell))) {
      boxes.push([y, x, cell]);
    }
  }));
  return {robot, boxes};
}

function signature(state) {
  const boxes = state.boxes.map(box => box.join(",")).sort().join(";");
  return `${state.robot.join(",")}|${boxes}`;
}

function unprunedNeighbors(state, board) {
  const occupied = new Map(state.boxes.map((box, index) => [`${box[0]},${box[1]}`, index]));
  const result = [];
  for (const [move, [dy, dx]] of Object.entries(DIRS)) {
    const [y, x] = state.robot;
    const next = `${y + dy},${x + dx}`;
    if (!board.floor.has(next)) continue;
    let boxes = state.boxes;
    let pushed = null;
    if (occupied.has(next)) {
      const destination = `${y + 2 * dy},${x + 2 * dx}`;
      if (!board.floor.has(destination) || occupied.has(destination)) continue;
      const index = occupied.get(next);
      boxes = state.boxes.map((box, boxIndex) =>
        boxIndex === index ? [y + 2 * dy, x + 2 * dx, box[2]] : box);
      pushed = {label: state.boxes[index][2], destination: [y + 2 * dy, x + 2 * dx]};
    }
    result.push({state: {robot: [y + dy, x + dx], boxes}, move, pushed});
  }
  return result;
}

function enumerateReachable(rows, limit = 50000) {
  const worker = loadWorker();
  const board = worker.parse({rows});
  const initial = stateFromRows(rows);
  const initialSignature = signature(initial);
  const states = new Map([[initialSignature, initial]]);
  const edges = new Map();
  const reverse = new Map();
  const queue = [initialSignature];

  for (let head = 0; head < queue.length; head++) {
    const parentSignature = queue[head];
    const outgoing = unprunedNeighbors(states.get(parentSignature), board);
    edges.set(parentSignature, outgoing);
    for (const edge of outgoing) {
      const childSignature = signature(edge.state);
      edge.signature = childSignature;
      if (!reverse.has(childSignature)) reverse.set(childSignature, new Set());
      reverse.get(childSignature).add(parentSignature);
      if (states.has(childSignature)) continue;
      assert.ok(states.size < limit, `state limit exceeded for board:\n${rows.join("\n")}`);
      states.set(childSignature, edge.state);
      queue.push(childSignature);
    }
  }

  const solvable = new Set();
  const solvedQueue = [];
  for (const [stateSignature, state] of states) {
    if (!worker.goal(state.boxes, board.goals)) continue;
    solvable.add(stateSignature);
    solvedQueue.push(stateSignature);
  }
  for (let head = 0; head < solvedQueue.length; head++) {
    for (const parent of reverse.get(solvedQueue[head]) || []) {
      if (solvable.has(parent)) continue;
      solvable.add(parent);
      solvedQueue.push(parent);
    }
  }
  return {worker, board, states, edges, solvable};
}

const DIFFERENTIAL_BOARDS = [
  [
    "OOOOOO",
    "O S  O",
    "O X  O",
    "O R  O",
    "OOOOOO",
  ],
  [
    "OOOOOOO",
    "O SS  O",
    "O XX  O",
    "O  R  O",
    "OOOOOOO",
  ],
  [
    "OOOOOOO",
    "O ab  O",
    "O AB  O",
    "O  R  O",
    "OOOOOOO",
  ],
  [
    "OOOOOOO",
    "O S   O",
    "O  OO O",
    "O X   O",
    "O     O",
    "O R   O",
    "OOOOOOO",
  ],
];

function generatedSolvableBoards() {
  const boards = [];
  for (const width of [6, 7, 8]) {
    for (const height of [5, 6]) {
      for (let column = 2; column <= width - 3; column++) {
        const rows = Array.from({length: height}, (_, y) =>
          y === 0 || y === height - 1 ? "O".repeat(width) : `O${" ".repeat(width - 2)}O`);
        const replace = (row, cell) => {
          rows[row] = rows[row].slice(0, column) + cell + rows[row].slice(column + 1);
        };
        replace(1, "S");
        replace(2, "X");
        replace(height - 2, "R");
        boards.push(rows);
      }
    }
  }
  return boards;
}

test("hard pruning never rejects an exhaustively proven solvable small state", () => {
  let checkedStates = 0;
  let checkedPushes = 0;
  const generatedBoards = generatedSolvableBoards();
  assert.ok(generatedBoards.length >= 10);
  for (const rows of [...DIFFERENTIAL_BOARDS, ...generatedBoards]) {
    const {worker, board, states, edges, solvable} = enumerateReachable(rows);
    assert.ok(solvable.size > 0, `test board has no solvable reachable states:\n${rows.join("\n")}`);

    for (const stateSignature of solvable) {
      const state = states.get(stateSignature);
      checkedStates++;
      if (!worker.goal(state.boxes, board.goals)) {
        const reachable = worker.reachablePaths(state, board);
        assert.equal(
          worker.createsSealedCorralDeadlock(state, board, reachable),
          false,
          `sealed-corral false positive at ${stateSignature}`,
        );
      }

      const retainedMoves = new Set(worker.neighbors(state, board).map(next => next.move));
      for (const edge of edges.get(stateSignature) || []) {
        if (!edge.pushed || !solvable.has(edge.signature)) continue;
        checkedPushes++;
        const [y, x] = edge.pushed.destination;
        const context = `${edge.move} from ${stateSignature} to ${edge.signature}`;
        assert.equal(worker.staticDead(y, x, board, edge.pushed.label), false,
          `static-dead false positive: ${context}`);
        assert.equal(worker.creates2x2Deadlock(edge.state.boxes, board, [y, x]), false,
          `2x2 false positive: ${context}`);
        assert.equal(worker.createsFrozenComponentDeadlock(edge.state.boxes, board, [y, x]), false,
          `freeze false positive: ${context}`);
        assert.equal(worker.createsDynamicDeadlock(edge.state.boxes, board, [y, x]), false,
          `dynamic false positive: ${context}`);
        assert.equal(retainedMoves.has(edge.move), true,
          `combined neighbor pruning false positive: ${context}`);
      }
    }
  }
  assert.ok(checkedStates >= 100, `expected broad state coverage, got ${checkedStates}`);
  assert.ok(checkedPushes >= 20, `expected broad push coverage, got ${checkedPushes}`);
});
