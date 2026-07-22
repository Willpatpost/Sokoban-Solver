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

function enumerateReachable(rows, worker = loadWorker(), limit = 50000) {
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
  [
    "OOOOOOOO",
    "O O    O",
    "O XS O O",
    "O  O X O",
    "O    O O",
    "O R S  O",
    "OOOOOOOO",
  ],
];

function connectedFloor(cells) {
  const cellSet = new Set(cells.map(([y, x]) => `${y},${x}`));
  if (!cellSet.size) return false;
  const reached = new Set([cellSet.values().next().value]);
  const queue = [...reached];
  for (let head = 0; head < queue.length; head++) {
    const [y, x] = queue[head].split(",").map(Number);
    for (const [dy, dx] of Object.values(DIRS)) {
      const next = `${y + dy},${x + dx}`;
      if (!cellSet.has(next) || reached.has(next)) continue;
      reached.add(next);
      queue.push(next);
    }
  }
  return reached.size === cellSet.size;
}

function exhaustiveTinyBoards() {
  const interior = [];
  for (let y = 1; y <= 2; y++) {
    for (let x = 1; x <= 3; x++) interior.push([y, x]);
  }
  const boards = [];
  for (let mask = 0; mask < (1 << interior.length); mask++) {
    const floor = interior.filter((_, index) => mask & (1 << index));
    if (floor.length < 3 || !connectedFloor(floor)) continue;
    for (let goal = 0; goal < floor.length; goal++) {
      for (let box = 0; box < floor.length; box++) {
        if (box === goal) continue;
        for (let robot = 0; robot < floor.length; robot++) {
          if (robot === goal || robot === box) continue;
          const rows = Array.from({length: 4}, () => [..."OOOOO"]);
          floor.forEach(([y, x]) => { rows[y][x] = " "; });
          rows[floor[goal][0]][floor[goal][1]] = "S";
          rows[floor[box][0]][floor[box][1]] = "X";
          rows[floor[robot][0]][floor[robot][1]] = "R";
          boards.push(rows.map(row => row.join("")));
        }
      }
    }
  }
  return boards;
}

function retainSmallestCounterexample(current, candidate) {
  if (!current) return candidate;
  const rank = value => [
    value.floorCells,
    value.boxes,
    value.rows.length * value.rows[0].length,
    value.state.length,
    value.rule,
    value.move || "",
  ];
  const left = rank(candidate), right = rank(current);
  for (let index = 0; index < left.length; index++) {
    if (left[index] < right[index]) return candidate;
    if (left[index] > right[index]) return current;
  }
  return current;
}

test("counterexample retention prefers the smallest reproducible board and state", () => {
  const larger = {
    rule: "dynamic", rows: ["OOOOOO", "ORX SO", "OOOOOO"],
    floorCells: 4, boxes: 1, state: "1,1|1,2,X", move: "Right",
  };
  const smaller = {
    rule: "static-dead", rows: ["OOOOO", "ORXSO", "OOOOO"],
    floorCells: 3, boxes: 1, state: "1,1|1,2,X", move: "Right",
  };
  assert.equal(retainSmallestCounterexample(larger, smaller), smaller);
  assert.equal(retainSmallestCounterexample(smaller, larger), smaller);
});

test("generated wall-ended closed diagonals are exhaustively unsolvable", () => {
  const slots = [[2, 2], [2, 4], [3, 3], [3, 5]];
  let checked = 0;
  for (let mask = 0; mask < (1 << slots.length); mask++) {
    if (slots.filter((_, index) => mask & (1 << index)).length !== 2) continue;
    const grid = Array.from({length: 7}, (_, y) =>
      [...(y === 0 || y === 6 ? "OOOOOOOO" : "O      O")]);
    grid[1][2] = "O";
    grid[4][5] = "O";
    slots.forEach(([y, x], index) => { grid[y][x] = mask & (1 << index) ? "X" : "O"; });
    grid[5][2] = "R";
    grid[5][3] = "S";
    grid[5][4] = "S";
    const base = grid.map(row => row.join(""));
    const variants = [base, base.map(row => [...row].reverse().join(""))];
    for (const rows of variants) {
      const {worker, board, states, solvable} = enumerateReachable(rows);
      const initial = stateFromRows(rows);
      const detected = initial.boxes.some(([y, x]) =>
        worker.createsClosedDiagonalDeadlock(initial.boxes, board, [y, x]));
      if (detected) assert.equal(solvable.size, 0, `false positive:\n${rows.join("\n")}`);
      assert.ok(states.size > 0);
      if (detected) checked++;
    }
  }
  assert.equal(checked, 2);
});

test("hard pruning never rejects an exhaustively proven solvable small state", () => {
  let checkedStates = 0, checkedPushes = 0, solvableTinyLayouts = 0;
  let smallestCounterexample = null;
  const worker = loadWorker();
  const generatedBoards = exhaustiveTinyBoards();
  assert.ok(generatedBoards.length >= 750);
  const cases = [
    ...DIFFERENTIAL_BOARDS.map(rows => ({rows, requiredSolvable: true})),
    ...generatedBoards.map(rows => ({rows, requiredSolvable: false})),
  ];
  for (const {rows, requiredSolvable} of cases) {
    const {board, states, edges, solvable} = enumerateReachable(rows, worker);
    if (requiredSolvable) {
      assert.ok(solvable.size > 0, `test board has no solvable reachable states:\n${rows.join("\n")}`);
    } else if (solvable.size) {
      solvableTinyLayouts++;
    }
    if (!solvable.size) continue;

    const record = (rule, state, edge = null) => {
      smallestCounterexample = retainSmallestCounterexample(smallestCounterexample, {
        rule,
        rows,
        floorCells: board.floor.size,
        boxes: state.boxes.length,
        state: signature(state),
        move: edge?.move || null,
        nextState: edge?.signature || null,
      });
    };

    for (const stateSignature of solvable) {
      const state = states.get(stateSignature);
      checkedStates++;
      let ordinaryPushes = null, lockedPushes = null;
      if (!worker.goal(state.boxes, board.goals)) {
        const reachable = worker.reachablePaths(state, board);
        if (worker.createsSealedCorralDeadlock(state, board, reachable)) {
          record("sealed-corral", state);
        }
        const pushKey = next => `${next.pushedFrom}>${next.pushedTo}`;
        ordinaryPushes = new Set(worker.pushNeighbors(state, board, reachable).map(pushKey));
        lockedPushes = new Set(worker.pushNeighbors(
          state,
          board,
          reachable,
          {lockProven: true},
        ).map(pushKey));
      }

      const retainedMoves = new Set(worker.neighbors(state, board).map(next => next.move));
      for (const edge of edges.get(stateSignature) || []) {
        if (!edge.pushed || !solvable.has(edge.signature)) continue;
        checkedPushes++;
        const [y, x] = edge.pushed.destination;
        if (worker.staticDead(y, x, board, edge.pushed.label)) record("static-dead", state, edge);
        if (worker.creates2x2Deadlock(edge.state.boxes, board, [y, x])) {
          record("2x2", state, edge);
        }
        if (worker.createsFrozenComponentDeadlock(edge.state.boxes, board, [y, x])) {
          record("freeze", state, edge);
        }
        if (worker.createsClosedDiagonalDeadlock(edge.state.boxes, board, [y, x])) {
          record("closed-diagonal", state, edge);
        }
        if (worker.createsDynamicDeadlock(edge.state.boxes, board, [y, x])) {
          record("dynamic", state, edge);
        }
        if (!retainedMoves.has(edge.move)) record("combined-neighbor", state, edge);
        const push = `${edge.state.robot.join(",")}>${edge.pushed.destination.join(",")}`;
        if (ordinaryPushes?.has(push) && !lockedPushes.has(push)) {
          record("proven-commitment", state, edge);
        }
      }
    }
  }
  if (smallestCounterexample) {
    assert.fail(
      `hard pruning rejected a solvable transition; smallest counterexample:\n` +
      `${JSON.stringify(smallestCounterexample, null, 2)}\n` +
      smallestCounterexample.rows.join("\n"),
    );
  }
  assert.ok(solvableTinyLayouts >= 50, `expected exhaustive solvable layouts, got ${solvableTinyLayouts}`);
  assert.ok(checkedStates >= 100, `expected broad state coverage, got ${checkedStates}`);
  assert.ok(checkedPushes >= 20, `expected broad push coverage, got ${checkedPushes}`);
});
