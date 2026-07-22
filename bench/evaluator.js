"use strict";

const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const MOVE_VECTORS = {Up: [-1, 0], Down: [1, 0], Left: [0, -1], Right: [0, 1]};
const pkey = (y, x) => `${y},${x}`;

function parseRows(rows) {
  const floor = new Set(), goalsByLabel = new Map(), boxes = new Map();
  let robot = null;
  rows.forEach((row, y) => [...row].forEach((cell, x) => {
    const position = pkey(y, x);
    if (cell !== "O") floor.add(position);
    if (cell === "R") robot = [y, x];
    if (cell === "X" || (/[A-Z]/.test(cell) && !"ORS".includes(cell))) boxes.set(position, cell);
    const goalLabel = cell === "S" ? "X" : /[a-z]/.test(cell) ? cell.toUpperCase() : null;
    if (goalLabel) {
      if (!goalsByLabel.has(goalLabel)) goalsByLabel.set(goalLabel, []);
      goalsByLabel.get(goalLabel).push(position);
    }
  }));
  return {rows, floor, goalsByLabel, robot, boxes};
}

function cloneState(state) {
  return {...state, robot: [...state.robot], boxes: new Map(state.boxes)};
}

function replayPath(rows, path) {
  const state = parseRows(rows);
  if (!state.robot || !Array.isArray(path)) return {valid: false, reason: "invalid-path"};
  let replay = cloneState(state), pushes = 0;
  for (let index = 0; index < path.length; index++) {
    const vector = MOVE_VECTORS[path[index]];
    if (!vector) return {valid: false, reason: `unknown-move-${path[index]}`, index};
    const [dy, dx] = vector, [y, x] = replay.robot;
    const next = pkey(y + dy, x + dx);
    if (!replay.floor.has(next)) return {valid: false, reason: "wall", index};
    if (replay.boxes.has(next)) {
      const beyond = pkey(y + 2 * dy, x + 2 * dx);
      if (!replay.floor.has(beyond) || replay.boxes.has(beyond)) {
        return {valid: false, reason: "blocked-push", index};
      }
      const label = replay.boxes.get(next);
      replay.boxes.delete(next);
      replay.boxes.set(beyond, label);
      pushes++;
    }
    replay.robot = [y + dy, x + dx];
  }
  return {valid: true, state: replay, moves: path.length, pushes};
}

function reversePushDistances(floor, goal) {
  const distances = new Map([[goal, 0]]), queue = [goal];
  for (let head = 0; head < queue.length; head++) {
    const [y, x] = queue[head].split(",").map(Number);
    for (const [dy, dx] of DIRS) {
      const previous = pkey(y - dy, x - dx);
      const support = pkey(y - 2 * dy, x - 2 * dx);
      if (!floor.has(previous) || !floor.has(support) || distances.has(previous)) continue;
      distances.set(previous, distances.get(queue[head]) + 1);
      queue.push(previous);
    }
  }
  return distances;
}

function minimumAssignmentCost(costs) {
  const size = costs.length;
  if (!size) return 0;
  if (costs.some(row => row.length !== size || row.every(cost => !Number.isFinite(cost)))) {
    return Infinity;
  }
  const blocked = 1e9;
  const rowPotential = Array(size + 1).fill(0), columnPotential = Array(size + 1).fill(0);
  const matching = Array(size + 1).fill(0), predecessor = Array(size + 1).fill(0);
  for (let row = 1; row <= size; row++) {
    matching[0] = row;
    const minimum = Array(size + 1).fill(blocked), used = Array(size + 1).fill(false);
    let column = 0;
    do {
      used[column] = true;
      const matchedRow = matching[column];
      let delta = blocked, nextColumn = 0;
      for (let candidate = 1; candidate <= size; candidate++) {
        if (used[candidate]) continue;
        const cost = costs[matchedRow - 1][candidate - 1];
        const reduced = (Number.isFinite(cost) ? cost : blocked) -
          rowPotential[matchedRow] - columnPotential[candidate];
        if (reduced < minimum[candidate]) {
          minimum[candidate] = reduced;
          predecessor[candidate] = column;
        }
        if (minimum[candidate] < delta) {
          delta = minimum[candidate];
          nextColumn = candidate;
        }
      }
      if (delta >= blocked) return Infinity;
      for (let candidate = 0; candidate <= size; candidate++) {
        if (used[candidate]) {
          rowPotential[matching[candidate]] += delta;
          columnPotential[candidate] -= delta;
        } else {
          minimum[candidate] -= delta;
        }
      }
      column = nextColumn;
    } while (matching[column] !== 0);
    do {
      const previous = predecessor[column];
      matching[column] = matching[previous];
      column = previous;
    } while (column !== 0);
  }
  let total = 0;
  for (let column = 1; column <= size; column++) {
    const cost = costs[matching[column] - 1][column - 1];
    if (!Number.isFinite(cost)) return Infinity;
    total += cost;
  }
  return total;
}

function remainingPushLowerBound(state) {
  const boxesByLabel = new Map();
  for (const [position, label] of state.boxes) {
    if (!boxesByLabel.has(label)) boxesByLabel.set(label, []);
    boxesByLabel.get(label).push(position);
  }
  let total = 0;
  for (const [label, boxes] of boxesByLabel) {
    const goals = state.goalsByLabel.get(label) || [];
    const distances = goals.map(goal => reversePushDistances(state.floor, goal));
    const costs = boxes.map(box => distances.map(table => table.get(box) ?? Infinity));
    total += minimumAssignmentCost(costs);
  }
  return total;
}

function checkpointMatchesReplay(checkpoint, replay) {
  if (!checkpoint.state) return true;
  if (checkpoint.state.robot && checkpoint.state.robot.join(",") !== replay.robot.join(",")) return false;
  if (!Array.isArray(checkpoint.state.boxes)) return true;
  const expected = checkpoint.state.boxes.map(([position, label]) => `${position},${label}`).sort();
  const actual = [...replay.boxes].map(([position, label]) => `${position},${label}`).sort();
  return expected.length === actual.length && expected.every((value, index) => value === actual[index]);
}

function evaluateCheckpoints(rows, result) {
  const candidates = [result.checkpoint, result.phaseCheckpoint, ...(result.checkpoints || [])]
    .filter(Boolean);
  const seen = new Set(), evaluations = [];
  let rejected = 0;
  for (const checkpoint of candidates) {
    if (!Array.isArray(checkpoint.path)) {
      rejected++;
      continue;
    }
    const pathKey = checkpoint.path.join(",");
    if (seen.has(pathKey)) continue;
    seen.add(pathKey);
    const replay = replayPath(rows, checkpoint.path);
    if (!replay.valid || !checkpointMatchesReplay(checkpoint, replay.state)) {
      rejected++;
      continue;
    }
    const remainingPushes = remainingPushLowerBound(replay.state);
    if (!Number.isFinite(remainingPushes)) {
      rejected++;
      continue;
    }
    evaluations.push({
      moves: replay.moves,
      pushes: replay.pushes,
      remainingPushes,
      projectedPushes: replay.pushes + remainingPushes,
    });
  }
  evaluations.sort((left, right) =>
    left.projectedPushes - right.projectedPushes ||
    left.remainingPushes - right.remainingPushes ||
    left.moves - right.moves);
  return {
    candidates: candidates.length,
    replayValid: evaluations.length,
    rejected,
    best: evaluations[0] || null,
  };
}

module.exports = {
  evaluateCheckpoints,
  minimumAssignmentCost,
  parseRows,
  remainingPushLowerBound,
  replayPath,
};
