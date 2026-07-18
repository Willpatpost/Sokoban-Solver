const DIRS = {Up: [-1, 0], Down: [1, 0], Left: [0, -1], Right: [0, 1]};
const OPPOSITE = {Up: "Down", Down: "Up", Left: "Right", Right: "Left"};
const MOVE_CODE = {Up: "U", Down: "D", Left: "L", Right: "R"};
const key = (robot, boxes) => `${robot.join(",")}|${[...boxes].sort().join(";")}`;
const pkey = (y, x) => `${y},${x}`;
const boxSignature = (boxes) => boxes.map(box => box.join(",")).sort().join(";");
const exactPushKey = state => `${state.robot.join(",")}|${boxSignature(state.boxes)}`;
const HEURISTIC_MEMO_LIMIT = 20000;
const DEADLOCK_MEMO_LIMIT = 10000;

function memoizeBounded(memo, key, value, limit = HEURISTIC_MEMO_LIMIT) {
  if (memo.size >= limit) memo.delete(memo.keys().next().value);
  memo.set(key, value);
  return value;
}

class BoundedDepthMap {
  constructor(limit) {
    this.limit = limit;
    this.values = new Map();
  }
  get(key) { return this.values.get(key); }
  has(key) { return this.values.has(key); }
  set(key, value) {
    if (this.values.has(key)) this.values.delete(key);
    this.values.set(key, value);
    while (this.values.size > this.limit) this.values.delete(this.values.keys().next().value);
  }
  get size() { return this.values.size; }
}

class Heap {
  constructor() { this.items = []; }
  push(item) {
    const a = this.items; a.push(item); let i = a.length - 1;
    while (i) { const p = (i - 1) >> 1; if (a[p][0] <= item[0]) break; a[i] = a[p]; i = p; }
    a[i] = item;
  }
  pop() {
    const a = this.items, root = a[0], last = a.pop();
    if (a.length) {
      let i = 0;
      while (true) {
        let c = i * 2 + 1; if (c >= a.length) break;
        if (c + 1 < a.length && a[c + 1][0] < a[c][0]) c++;
        if (a[c][0] >= last[0]) break; a[i] = a[c]; i = c;
      }
      a[i] = last;
    }
    return root;
  }
  get length() { return this.items.length; }
}

function floorNeighbors(position, floor) {
  const [y, x] = position.split(",").map(Number);
  return Object.values(DIRS).map(([dy, dx]) => pkey(y + dy, x + dx))
    .filter(next => floor.has(next));
}

function floorComponents(floor, blocked = null) {
  const remaining = new Set(floor);
  if (blocked) remaining.delete(blocked);
  const components = [];
  while (remaining.size) {
    const start = remaining.values().next().value;
    const component = new Set([start]), queue = [start];
    remaining.delete(start);
    for (let head = 0; head < queue.length; head++) {
      for (const next of floorNeighbors(queue[head], floor)) {
        if (next === blocked || !remaining.has(next)) continue;
        remaining.delete(next);
        component.add(next);
        queue.push(next);
      }
    }
    components.push(component);
  }
  return components;
}

function articulationPoints(floor) {
  const discovered = new Map(), low = new Map(), parent = new Map(), result = new Set();
  let time = 0;
  const visit = position => {
    discovered.set(position, ++time);
    low.set(position, time);
    let children = 0;
    for (const next of floorNeighbors(position, floor)) {
      if (!discovered.has(next)) {
        parent.set(next, position);
        children++;
        visit(next);
        low.set(position, Math.min(low.get(position), low.get(next)));
        if (!parent.has(position) && children > 1) result.add(position);
        if (parent.has(position) && low.get(next) >= discovered.get(position)) result.add(position);
      } else if (next !== parent.get(position)) {
        low.set(position, Math.min(low.get(position), discovered.get(next)));
      }
    }
  };
  for (const position of floor) if (!discovered.has(position)) visit(position);
  return result;
}

function analyzeTopology(floor, goals) {
  const articulations = articulationPoints(floor);
  const tunnels = new Set([...floor].filter(position => {
    const neighbors = floorNeighbors(position, floor);
    if (neighbors.length !== 2) return false;
    const coordinates = neighbors.map(next => next.split(",").map(Number));
    return coordinates[0][0] === coordinates[1][0] || coordinates[0][1] === coordinates[1][1];
  }));
  const candidates = [];
  for (const gate of articulations) {
    const components = floorComponents(floor, gate).sort((left, right) => right.size - left.size);
    for (const cells of components) {
      if (cells.size < 2 || cells.size > floor.size * 0.72) continue;
      const roomGoals = [...goals.keys()].filter(goal => cells.has(goal));
      if (!roomGoals.length) continue;
      const depths = new Map(), queue = [];
      for (const neighbor of floorNeighbors(gate, floor)) {
        if (!cells.has(neighbor)) continue;
        depths.set(neighbor, 1);
        queue.push(neighbor);
      }
      for (let head = 0; head < queue.length; head++) {
        const position = queue[head], distance = depths.get(position);
        for (const next of floorNeighbors(position, floor)) {
          if (!cells.has(next) || depths.has(next)) continue;
          depths.set(next, distance + 1);
          queue.push(next);
        }
      }
      const traffic = new Map([...cells].map(cell => [cell, 0]));
      for (const goal of roomGoals) {
        const goalDistances = new Map([[goal, 0]]), goalQueue = [goal];
        for (let head = 0; head < goalQueue.length; head++) {
          const position = goalQueue[head], distance = goalDistances.get(position);
          for (const next of floorNeighbors(position, floor)) {
            if (!cells.has(next) || goalDistances.has(next)) continue;
            goalDistances.set(next, distance + 1);
            goalQueue.push(next);
          }
        }
        const goalDepth = depths.get(goal);
        for (const cell of cells) {
          if ((depths.get(cell) ?? Infinity) + (goalDistances.get(cell) ?? Infinity) === goalDepth) {
            traffic.set(cell, traffic.get(cell) + 1);
          }
        }
      }
      const dependencies = [];
      for (const blocker of roomGoals) {
        const reducedFloor = new Set(floor);
        reducedFloor.delete(blocker);
        for (const target of roomGoals) {
          if (target === blocker) continue;
          const normallyReachable = reversePushDistances(floor, target).has(gate);
          const blockedReachable = reversePushDistances(reducedFloor, target).has(gate);
          if (normallyReachable && !blockedReachable) dependencies.push([blocker, target]);
        }
      }
      candidates.push({gate, cells, goals: roomGoals, depths, traffic, dependencies});
    }
  }
  candidates.sort((left, right) => right.cells.size - left.cells.size);
  const rooms = [];
  for (const candidate of candidates) {
    if (rooms.some(room => [...candidate.cells].every(cell => room.cells.has(cell)))) continue;
    rooms.push(candidate);
  }
  return {articulations, rooms, tunnels};
}

function parse(data) {
  const floor = new Set(), walls = new Set(), goals = new Map(), goalsByLabel = new Map();
  data.rows.forEach((row, y) => [...row].forEach((ch, x) => {
    const p = pkey(y, x);
    if (ch === "O") walls.add(p); else floor.add(p);
    const label = ch === "S" ? "X" : /[a-z]/.test(ch) ? ch.toUpperCase() : null;
    if (label) {
      goals.set(p, label);
      if (!goalsByLabel.has(label)) goalsByLabel.set(label, []);
      goalsByLabel.get(label).push(p);
    }
  }));
  const pushDistances = new Map([...goals.keys()].map(goal => [goal, reversePushDistances(floor, goal)]));
  const goalPressure = new Map([...goals.keys()].map(goal => [
    goal,
    floor.size / Math.max(1, pushDistances.get(goal).size),
  ]));
  const topology = analyzeTopology(floor, goals);
  return {
    rows: data.rows, floor, walls, goals, goalsByLabel, pushDistances, goalPressure,
    topology, heuristicMemo: new Map(), playerPushDistances: new Map(),
    deadlockMemo: new Map(),
  };
}
function reversePushDistances(floor, goalKey) {
  const [gy, gx] = goalKey.split(",").map(Number);
  const distances = new Map([[goalKey, 0]]), queue = [[gy, gx]];
  for (let head = 0; head < queue.length; head++) {
    const [y, x] = queue[head], distance = distances.get(pkey(y, x));
    for (const [dy, dx] of Object.values(DIRS)) {
      const previous = pkey(y - dy, x - dx);
      const support = pkey(y - 2 * dy, x - 2 * dx);
      if (!floor.has(previous) || !floor.has(support) || distances.has(previous)) continue;
      distances.set(previous, distance + 1);
      queue.push([y - dy, x - dx]);
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
  const rowPotential = Array(size + 1).fill(0);
  const columnPotential = Array(size + 1).fill(0);
  const matching = Array(size + 1).fill(0);
  const predecessor = Array(size + 1).fill(0);
  for (let row = 1; row <= size; row++) {
    matching[0] = row;
    const minimum = Array(size + 1).fill(blocked);
    const used = Array(size + 1).fill(false);
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

function singleBoxReachable(floor, boxKey, startKey) {
  const reached = new Set([startKey]), queue = [startKey];
  for (let head = 0; head < queue.length; head++) {
    const [y, x] = queue[head].split(",").map(Number);
    for (const [dy, dx] of Object.values(DIRS)) {
      const next = pkey(y + dy, x + dx);
      if (next === boxKey || !floor.has(next) || reached.has(next)) continue;
      reached.add(next);
      queue.push(next);
    }
  }
  return reached;
}

function playerAwarePushDistances(board, startKey) {
  if (board.playerPushDistances.has(startKey)) return board.playerPushDistances.get(startKey);
  const initialRegions = [], unassigned = new Set(board.floor);
  unassigned.delete(startKey);
  while (unassigned.size) {
    const representative = unassigned.values().next().value;
    const region = singleBoxReachable(board.floor, startKey, representative);
    region.forEach(position => unassigned.delete(position));
    initialRegions.push(representative);
  }

  const distances = new Map([[startKey, 0]]), seen = new Set(), queue = [];
  const enqueue = (boxKey, robotKey, distance) => {
    const region = singleBoxReachable(board.floor, boxKey, robotKey);
    let representative = null;
    for (const position of region) {
      if (representative === null || position < representative) representative = position;
    }
    const signature = `${boxKey}|${representative}`;
    if (seen.has(signature)) return;
    seen.add(signature);
    queue.push({boxKey, region, distance});
  };
  initialRegions.forEach(robotKey => enqueue(startKey, robotKey, 0));

  for (let head = 0; head < queue.length; head++) {
    const {boxKey, region, distance} = queue[head];
    const [y, x] = boxKey.split(",").map(Number);
    for (const [dy, dx] of Object.values(DIRS)) {
      const support = pkey(y - dy, x - dx), destination = pkey(y + dy, x + dx);
      if (!region.has(support) || !board.floor.has(destination)) continue;
      const nextDistance = distance + 1;
      if (nextDistance < (distances.get(destination) ?? Infinity)) {
        distances.set(destination, nextDistance);
      }
      enqueue(destination, boxKey, nextDistance);
    }
  }
  board.playerPushDistances.set(startKey, distances);
  return distances;
}

function heuristic(boxes, board) {
  const signature = boxSignature(boxes);
  if (board.heuristicMemo.has(signature)) return board.heuristicMemo.get(signature);
  const byLabel = new Map();
  boxes.forEach(([y, x, label]) => {
    if (!byLabel.has(label)) byLabel.set(label, []);
    byLabel.get(label).push([y, x]);
  });
  let total = 0;
  for (const [label, positions] of byLabel) {
    const targets = board.goalsByLabel.get(label) || [];
    const costs = positions.map(([y, x]) => {
      const distances = playerAwarePushDistances(board, pkey(y, x));
      return targets.map(target => distances.get(target) ?? Infinity);
    });
    total += minimumAssignmentCost(costs);
  }
  return memoizeBounded(board.heuristicMemo, signature, total);
}

function topologyPenalty(boxes, board) {
  const occupied = new Map(boxes.map(([y, x, label]) => [pkey(y, x), label]));
  let penalty = 0;
  for (const room of board.topology.rooms) {
    const boxesInside = boxes.filter(([y, x]) => room.cells.has(pkey(y, x)));
    penalty += Math.abs(boxesInside.length - room.goals.length);
    for (const [y, x, label] of boxesInside) {
      const position = pkey(y, x);
      if (board.goals.get(position) !== label) penalty += 0.35 * (room.traffic.get(position) || 0);
    }

    const unsolved = room.goals.filter(goal => occupied.get(goal) !== board.goals.get(goal));
    for (const goal of room.goals) {
      if (occupied.get(goal) !== board.goals.get(goal)) continue;
      const depth = room.depths.get(goal) || 0;
      for (const pending of unsolved) {
        const pendingDepth = room.depths.get(pending) || 0;
        if (pendingDepth > depth) penalty += 1 + pendingDepth - depth;
      }
    }
    for (const [blocker, prerequisite] of room.dependencies) {
      const blockerSolved = occupied.get(blocker) === board.goals.get(blocker);
      const prerequisiteSolved = occupied.get(prerequisite) === board.goals.get(prerequisite);
      if (blockerSolved && !prerequisiteSolved) penalty += 4;
    }
    if (occupied.has(room.gate) && unsolved.length) penalty += 2 * unsolved.length;
  }
  return penalty;
}
function forwardPushDistances(floor, startKey) {
  const [startY, startX] = startKey.split(",").map(Number);
  const distances = new Map([[startKey, 0]]), queue = [[startY, startX]];
  for (let head = 0; head < queue.length; head++) {
    const [y, x] = queue[head], distance = distances.get(pkey(y, x));
    for (const [dy, dx] of Object.values(DIRS)) {
      const destination = pkey(y + dy, x + dx);
      const support = pkey(y - dy, x - dx);
      if (!floor.has(destination) || !floor.has(support) || distances.has(destination)) continue;
      distances.set(destination, distance + 1);
      queue.push([y + dy, x + dx]);
    }
  }
  return distances;
}
function targetMapFromBoxes(boxes, floor) {
  const targets = new Map();
  boxes.forEach(([y, x, label]) => {
    if (!targets.has(label)) targets.set(label, []);
    targets.get(label).push({distances: forwardPushDistances(floor, pkey(y, x))});
  });
  targets.memo = new Map();
  return targets;
}
function homeHeuristic(boxes, targetsByLabel) {
  const signature = boxSignature(boxes);
  if (targetsByLabel.memo.has(signature)) return targetsByLabel.memo.get(signature);
  const byLabel = new Map();
  boxes.forEach(([y, x, label]) => {
    if (!byLabel.has(label)) byLabel.set(label, []);
    byLabel.get(label).push([y, x]);
  });
  let total = 0;
  for (const [label, positions] of byLabel) {
    const targets = targetsByLabel.get(label) || [];
    const costs = positions.map(([y, x]) => targets.map(target => (
      target.distances.get(pkey(y, x)) ?? Infinity
    )));
    total += minimumAssignmentCost(costs);
  }
  return memoizeBounded(targetsByLabel.memo, signature, total);
}
function goal(boxes, goals) {
  return boxes.every(([y, x, label]) => goals.get(pkey(y, x)) === label);
}
function goalPackingBonus(boxes, board) {
  return boxes.reduce((bonus, [y, x, label]) => {
    const position = pkey(y, x);
    return bonus + (board.goals.get(position) === label ? board.goalPressure.get(position) || 0 : 0);
  }, 0);
}
function corner(y, x, board, label) {
  if (board.goals.get(pkey(y, x)) === label) return false;
  const wall = (dy, dx) => board.walls.has(pkey(y + dy, x + dx));
  return (wall(-1, 0) && wall(0, -1)) || (wall(-1, 0) && wall(0, 1)) ||
    (wall(1, 0) && wall(0, -1)) || (wall(1, 0) && wall(0, 1));
}
function staticDead(y, x, board, label) {
  if (board.goals.get(pkey(y, x)) === label) return false;
  const distances = playerAwarePushDistances(board, pkey(y, x));
  return !(board.goalsByLabel.get(label) || []).some(goal => distances.has(goal));
}
function creates2x2Deadlock(boxes, board, movedBox) {
  const occupied = new Map(boxes.map(([y, x, label]) => [pkey(y, x), label]));
  const [boxY, boxX] = movedBox;
  for (const originY of [boxY - 1, boxY]) {
    for (const originX of [boxX - 1, boxX]) {
      const cells = [
        [originY, originX], [originY + 1, originX],
        [originY, originX + 1], [originY + 1, originX + 1],
      ];
      if (!cells.every(([y, x]) => board.walls.has(pkey(y, x)) || occupied.has(pkey(y, x)))) continue;
      if (cells.some(([y, x]) => {
        const label = occupied.get(pkey(y, x));
        return label && board.goals.get(pkey(y, x)) !== label;
      })) return true;
    }
  }
  return false;
}
function createsFrozenComponentDeadlock(boxes, board, movedBox) {
  const occupied = new Map(boxes.map(([y, x, label]) => [pkey(y, x), label]));
  const start = pkey(movedBox[0], movedBox[1]);
  if (!occupied.has(start)) return false;
  const component = new Set([start]), queue = [movedBox];
  for (let head = 0; head < queue.length; head++) {
    const [y, x] = queue[head];
    for (const [dy, dx] of Object.values(DIRS)) {
      const adjacent = pkey(y + dy, x + dx);
      if (!occupied.has(adjacent) || component.has(adjacent)) continue;
      component.add(adjacent);
      queue.push([y + dy, x + dx]);
    }
  }
  const movable = queue.some(([y, x]) => Object.values(DIRS).some(([dy, dx]) => {
    const destination = pkey(y + dy, x + dx);
    const support = pkey(y - dy, x - dx);
    return board.floor.has(destination) && board.floor.has(support) &&
      !occupied.has(destination) && !occupied.has(support);
  }));
  if (movable) return false;
  return [...component].some(position => board.goals.get(position) !== occupied.get(position));
}

function createsDynamicDeadlock(boxes, board, movedBox) {
  const signature = `${boxSignature(boxes)}|${movedBox.join(",")}`;
  if (board.deadlockMemo.has(signature)) return board.deadlockMemo.get(signature);
  const deadlocked = creates2x2Deadlock(boxes, board, movedBox) ||
    createsFrozenComponentDeadlock(boxes, board, movedBox);
  return memoizeBounded(board.deadlockMemo, signature, deadlocked, DEADLOCK_MEMO_LIMIT);
}
function neighbors(state, board) {
  const occupied = new Map(state.boxes.map((b, i) => [pkey(b[0], b[1]), i])), result = [];
  for (const [move, [dy, dx]] of Object.entries(DIRS)) {
    const [y, x] = state.robot, ny = y + dy, nx = x + dx, next = pkey(ny, nx);
    if (!board.floor.has(next)) continue;
    let boxes = state.boxes;
    if (occupied.has(next)) {
      const by = ny + dy, bx = nx + dx, beyond = pkey(by, bx);
      if (!board.floor.has(beyond) || occupied.has(beyond)) continue;
      const index = occupied.get(next), label = boxes[index][2];
      boxes = boxes.map((b, i) => i === index ? [by, bx, label] : b);
      if (staticDead(by, bx, board, label) ||
          createsDynamicDeadlock(boxes, board, [by, bx])) continue;
    }
    result.push({robot: [ny, nx], boxes, move});
  }
  return result;
}

function reachablePaths(state, board) {
  const occupied = new Set(state.boxes.map(b => pkey(b[0], b[1])));
  const start = pkey(state.robot[0], state.robot[1]);
  const parents = new Map([[start, {parent: null, move: null}]]);
  const queue = [state.robot];
  for (let head = 0; head < queue.length; head++) {
    const [y, x] = queue[head], current = pkey(y, x);
    for (const [move, [dy, dx]] of Object.entries(DIRS)) {
      const ny = y + dy, nx = x + dx, next = pkey(ny, nx);
      if (parents.has(next) || !board.floor.has(next) || occupied.has(next)) continue;
      parents.set(next, {parent: current, move});
      queue.push([ny, nx]);
    }
  }
  return {
    has: (position) => parents.has(position),
    get: (position) => {
      const path = [];
      let current = position;
      while (parents.has(current) && parents.get(current).parent !== null) {
        const record = parents.get(current);
        path.unshift(record.move);
        current = record.parent;
      }
      return path;
    },
    keys: () => parents.keys(),
    size: parents.size,
  };
}

function createsSealedCorralDeadlock(state, board, reachable) {
  const occupied = new Map(state.boxes.map(([y, x, label]) => [pkey(y, x), label]));
  const inaccessible = new Set([...board.floor].filter(position => !reachable.has(position)));
  while (inaccessible.size) {
    const start = inaccessible.values().next().value;
    const component = new Set([start]), queue = [start];
    inaccessible.delete(start);
    for (let head = 0; head < queue.length; head++) {
      for (const next of floorNeighbors(queue[head], board.floor)) {
        if (!inaccessible.has(next)) continue;
        inaccessible.delete(next);
        component.add(next);
        queue.push(next);
      }
    }
    const componentBoxes = [...component].filter(position => occupied.has(position));
    if (!componentBoxes.some(position => board.goals.get(position) !== occupied.get(position))) continue;
    const canOpen = componentBoxes.some(position => {
      const [y, x] = position.split(",").map(Number);
      return Object.values(DIRS).some(([dy, dx]) => {
        const support = pkey(y - dy, x - dx), destination = pkey(y + dy, x + dx);
        return reachable.has(support) && board.floor.has(destination) && !occupied.has(destination);
      });
    });
    if (!canOpen) return true;
  }
  return false;
}

function pushNeighbors(state, board, reachable = reachablePaths(state, board)) {
  const occupied = new Map(state.boxes.map((b, i) => [pkey(b[0], b[1]), i]));
  const result = [];
  state.boxes.forEach(([y, x, label], index) => {
    for (const [move, [dy, dx]] of Object.entries(DIRS)) {
      const support = pkey(y - dy, x - dx), dest = pkey(y + dy, x + dx);
      if (!reachable.has(support) || !board.floor.has(dest) || occupied.has(dest)) continue;
      const boxes = state.boxes.map((b, i) => i === index ? [y + dy, x + dx, label] : b);
      if (staticDead(y + dy, x + dx, board, label) ||
          createsDynamicDeadlock(boxes, board, [y + dy, x + dx])) continue;
      result.push({
        robot: [y, x],
        boxes,
        path: [...reachable.get(support), move],
        pushClass: `${label}:${y},${x}:${move}`,
        pushedTo: dest,
      });
    }
  });
  return result;
}

function pushKey(state, reachable) {
  let robotRegion = null;
  for (const position of reachable.keys()) {
    if (robotRegion === null || position < robotRegion) robotRegion = position;
  }
  return `${robotRegion}|${[...state.boxes.map(b => b.join(","))].sort().join(";")}`;
}

function collapseForcedPushes(first, board, limit = 32) {
  let state = {robot: first.robot, boxes: first.boxes};
  const path = [...first.path], seen = new Set([exactPushKey(state)]);
  let pushes = 1;
  while (pushes < limit && !goal(state.boxes, board.goals)) {
    const reachable = reachablePaths(state, board);
    if (createsSealedCorralDeadlock(state, board, reachable)) return null;
    const choices = pushNeighbors(state, board, reachable);
    if (choices.length !== 1) break;
    const next = choices[0], signature = exactPushKey(next);
    if (seen.has(signature)) break;
    seen.add(signature);
    path.push(...next.path);
    state = {robot: next.robot, boxes: next.boxes};
    pushes++;
  }
  return {...state, path, pushes, pushClass: first.pushClass};
}

function expandPushMacro(next, board, enabled = true) {
  if (!enabled || !board.topology.tunnels.has(next.pushedTo)) return {...next, pushes: 1};
  return collapseForcedPushes(next, board);
}

function invertWalk(path) {
  return [...path].reverse().map(move => OPPOSITE[move]);
}
function encodeMoves(path) {
  return path.map(move => MOVE_CODE[move]).join("");
}

function reversePullNeighbors(state, board, reachable = reachablePaths(state, board)) {
  const occupied = new Map(state.boxes.map((b, i) => [pkey(b[0], b[1]), i]));
  const result = [];
  state.boxes.forEach(([y, x, label], index) => {
    for (const [move, [dy, dx]] of Object.entries(DIRS)) {
      const boxBefore = pkey(y - dy, x - dx);
      const robotAfterPull = pkey(y - 2 * dy, x - 2 * dx);
      if (!reachable.has(boxBefore) || !board.floor.has(robotAfterPull)) continue;
      if (occupied.has(boxBefore) || occupied.has(robotAfterPull)) continue;
      if (staticDead(y - dy, x - dx, board, label)) continue;
      const boxes = state.boxes.map((b, i) => i === index ? [y - dy, x - dx, label] : b);
      if (creates2x2Deadlock(boxes, board, [y - dy, x - dx]) ||
          createsFrozenComponentDeadlock(boxes, board, [y - dy, x - dx])) continue;
      const walkToPullSpot = reachable.get(boxBefore);
      const walkFromPushLanding = invertWalk(walkToPullSpot);
      result.push({
        robot: [y - 2 * dy, x - 2 * dx],
        boxes,
        cost: state.cost + 1,
        segment: [move, ...walkFromPushLanding],
      });
    }
  });
  return result;
}

function solvedBoxes(board, initialBoxes) {
  const byLabel = new Map();
  initialBoxes.forEach(([, , label]) => byLabel.set(label, (byLabel.get(label) || 0) + 1));
  const boxes = [];
  for (const [label, count] of byLabel) {
    const goals = [...board.goals]
      .filter(([, goalLabel]) => goalLabel === label)
      .map(([position]) => position.split(",").map(Number))
      .slice(0, count);
    goals.forEach(([y, x]) => boxes.push([y, x, label]));
  }
  return boxes.sort((a, b) => a.join(",").localeCompare(b.join(",")));
}

function reverseStartStates(board, initialBoxes, shard) {
  const boxes = solvedBoxes(board, initialBoxes);
  const occupied = new Set(boxes.map(([y, x]) => pkey(y, x)));
  const unique = new Map();
  for (const position of board.floor) {
    if (occupied.has(position)) continue;
    const robot = position.split(",").map(Number);
    const state = {robot, boxes, cost: 0};
    const reachable = reachablePaths(state, board);
    const signature = pushKey(state, reachable);
    if (!unique.has(signature)) unique.set(signature, state);
  }
  return [...unique.values()].filter((_state, index) => index % shard.count === shard.index);
}

function flushRecords(records) {
  if (records.length) postMessage({type: "records", records: records.splice(0, records.length)});
}

function reconstructPath(cameFrom, signature) {
  const path = [];
  let current = signature;
  while (cameFrom.has(current)) {
    const {parent, segment} = cameFrom.get(current);
    path.unshift(...segment);
    current = parent;
  }
  return path;
}

function signatureNoise(signature, seed) {
  let hash = (2166136261 ^ seed) >>> 0;
  for (let index = 0; index < signature.length; index++) {
    hash ^= signature.charCodeAt(index);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash / 0x100000000;
}

function reconstructNodePath(node) {
  const segments = [];
  for (let current = node; current; current = current.parent) segments.push(current.segment);
  const path = [];
  for (let index = segments.length - 1; index >= 0; index--) path.push(...segments[index]);
  return path;
}

function takeDiverse(candidates, count, selected, scoreKey) {
  const groups = new Map();
  for (const candidate of candidates) {
    if (selected.has(candidate.exactSignature)) continue;
    const key = candidate.pushClass || candidate.exactSignature;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(candidate);
  }
  let queues = [...groups.values()].map(items => ({
    items: items.sort((left, right) => left[scoreKey] - right[scoreKey]),
    index: 0,
  }));
  queues.sort((left, right) => left.items[0][scoreKey] - right.items[0][scoreKey]);
  const result = [];
  while (result.length < count && queues.length) {
    const remaining = [];
    for (const queue of queues) {
      if (result.length >= count) break;
      const candidate = queue.items[queue.index++];
      if (!selected.has(candidate.exactSignature)) {
        selected.add(candidate.exactSignature);
        result.push(candidate);
      }
      if (queue.index < queue.items.length) remaining.push(queue);
    }
    queues = remaining;
  }
  return result;
}

function selectBeamLayer(candidates, width, profile = "balanced") {
  if (candidates.length <= width) return candidates;
  let bestEstimate = Infinity;
  for (const candidate of candidates) bestEstimate = Math.min(bestEstimate, candidate.estimate);
  const bands = [[], [], [], []];
  for (const candidate of candidates) {
    const slack = candidate.estimate - bestEstimate;
    bands[slack <= 2 ? 0 : slack <= 5 ? 1 : slack <= 9 ? 2 : 3].push(candidate);
  }
  const ratios = profile === "detour" ? [0.30, 0.25, 0.25, 0.20] : [0.50, 0.25, 0.15, 0.10];
  const selected = new Set(), result = [];
  bands.forEach((band, index) => {
    const quota = index === bands.length - 1
      ? width - ratios.slice(0, index).reduce((total, ratio) => total + Math.floor(width * ratio), 0)
      : Math.floor(width * ratios[index]);
    const scoreKey = index === bands.length - 1 ? "exploreScore" : "score";
    result.push(...takeDiverse(band, quota, selected, scoreKey));
  });
  if (result.length < width) {
    const ranked = [...candidates].sort((left, right) => left.score - right.score);
    result.push(...takeDiverse(ranked, width - result.length, selected, "score"));
  }
  return result;
}

function beamSearch(payload) {
  const board = payload.preparedBoard || parse(payload.state);
  const initial = {
    robot: payload.state.robot,
    boxes: payload.state.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]),
    cost: 0,
  };
  const width = payload.beamWidth || 3000;
  const maxDepth = payload.maxDepth || 500;
  const weight = payload.weight || 3;
  const diversity = payload.diversity ?? 1.5;
  const goalPackingWeight = payload.goalPackingWeight ?? 0.8;
  const mobilityWeight = payload.mobilityWeight ?? 0.03;
  const topologyWeight = payload.topologyWeight ?? 0.7;
  const beamProfile = payload.beamProfile || "balanced";
  const seed = payload.seed || 0;
  const transpositionLimit = payload.transpositionLimit || Math.max(12000, width * 60);
  const seenDepth = new BoundedDepthMap(transpositionLimit);
  const seenExactDepth = new BoundedDepthMap(Math.max(8000, Math.floor(transpositionLimit / 2)));
  let visited = 0, reported = 0, bestEstimate = Infinity, bestPushes = 0;

  initial.reachable = reachablePaths(initial, board);
  if (createsSealedCorralDeadlock(initial, board, initial.reachable)) {
    return {path: null, visited};
  }
  initial.signature = pushKey(initial, initial.reachable);
  const initialEstimate = heuristic(initial.boxes, board);
  if (!Number.isFinite(initialEstimate)) return {path: null, visited};
  bestEstimate = initialEstimate;
  seenDepth.set(initial.signature, 0);
  let beam = [initial];

  for (let depth = 0; beam.length && depth <= maxDepth; depth++) {
    const candidates = new Map();
    for (const current of beam) {
      visited++;
      if (goal(current.boxes, board.goals)) {
        return {path: reconstructNodePath(current.node), visited};
      }
      if (payload.maxVisited && visited >= payload.maxVisited) {
        return {path: null, visited, cutoff: true, bestEstimate, bestPushes};
      }
      for (const rawNext of pushNeighbors(current, board, current.reachable)) {
        const next = expandPushMacro(rawNext, board, payload.forcedMacros !== false);
        if (!next) continue;
        const child = {robot: next.robot, boxes: next.boxes, cost: current.cost + next.pushes};
        if (child.cost > maxDepth) continue;
        if (payload.upperBound && child.cost > payload.upperBound) continue;
        if (goal(child.boxes, board.goals)) {
          return {
            path: [...reconstructNodePath(current.node), ...next.path],
            visited,
          };
        }
        child.exactSignature = exactPushKey(child);
        if ((seenExactDepth.get(child.exactSignature) ?? Infinity) <= child.cost) continue;
        const estimate = heuristic(child.boxes, board);
        if (!Number.isFinite(estimate)) continue;
        if (payload.upperBound && child.cost + estimate > payload.upperBound) continue;
        if (estimate < bestEstimate) {
          bestEstimate = estimate;
          bestPushes = child.cost;
        }
        const topology = topologyPenalty(child.boxes, board);
        const score = weight * estimate + topologyWeight * topology -
          goalPackingWeight * goalPackingBonus(child.boxes, board) +
          diversity * signatureNoise(child.exactSignature, seed);
        const exploreScore = topologyWeight * topology -
          goalPackingWeight * goalPackingBonus(child.boxes, board) +
          diversity * signatureNoise(child.exactSignature, seed + 7919);
        const existing = candidates.get(child.exactSignature);
        if (!existing || score < existing.score) {
          candidates.set(child.exactSignature, {
            ...child,
            node: {parent: current.node || null, segment: next.path},
            estimate,
            topology,
            score,
            exploreScore,
            pushClass: next.pushClass,
          });
        }
      }
      if (visited - reported >= 5000) {
        postMessage({type: "progress", visited: (payload.progressOffset || 0) + visited});
        reported = visited;
      }
    }
    const shortlist = selectBeamLayer([...candidates.values()], width * 3, beamProfile);
    beam = [];
    for (const child of shortlist) {
      child.reachable = reachablePaths(child, board);
      if (createsSealedCorralDeadlock(child, board, child.reachable)) continue;
      child.signature = pushKey(child, child.reachable);
      if ((seenDepth.get(child.signature) ?? Infinity) <= child.cost) continue;
      child.score -= mobilityWeight * child.reachable.size;
      child.exploreScore -= mobilityWeight * child.reachable.size;
      seenDepth.set(child.signature, child.cost);
      seenExactDepth.set(child.exactSignature, child.cost);
      beam.push(child);
    }
    beam = selectBeamLayer(beam, width, beamProfile);
  }
  return {path: null, visited, bestEstimate, bestPushes};
}

function beamRestartSearch(payload) {
  const restartCount = payload.restartCount || 3;
  const restartVisited = payload.restartVisited || 180000;
  const seedStride = payload.seedStride || 104729;
  const profiles = payload.restartProfiles?.length ? payload.restartProfiles : [{}];
  const preparedBoard = parse(payload.state);
  let visited = 0, bestEstimate = Infinity, bestPushes = 0;
  for (let restart = 0; restart < restartCount; restart++) {
    const result = beamSearch({
      ...payload,
      ...profiles[restart % profiles.length],
      algorithm: "push-beam",
      preparedBoard,
      maxVisited: restartVisited,
      progressOffset: visited,
      seed: (payload.seed || 0) + restart * seedStride,
    });
    visited += result.visited;
    if ((result.bestEstimate ?? Infinity) < bestEstimate) {
      bestEstimate = result.bestEstimate;
      bestPushes = result.bestPushes || 0;
    }
    if (result.path) return {...result, visited, restart: restart + 1};
  }
  return {path: null, visited, cutoff: true, bestEstimate, bestPushes, restarts: restartCount};
}

function bidirectionalSide(payload) {
  const board = parse(payload.state);
  const initialBoxes = payload.state.boxes.map(([p, label]) => [...p.split(",").map(Number), label]);
  const initialTargets = targetMapFromBoxes(initialBoxes, board.floor);
  const forward = payload.mode === "bidir-forward";
  const frontier = new Heap(), bestCost = new Map(), closed = new Set(), records = [];
  let order = 0, visited = 0, reported = 0;
  const starts = forward
    ? [{robot: payload.state.robot, boxes: initialBoxes, cost: 0, path: []}]
    : reverseStartStates(board, initialBoxes, payload.reverseShard || {index: 0, count: 1});
  starts.forEach(state => {
    state.exactSignature = exactPushKey(state);
    const estimate = forward
      ? heuristic(state.boxes, board)
      : homeHeuristic(state.boxes, initialTargets);
    if (!Number.isFinite(estimate) || bestCost.has(state.exactSignature)) return;
    if (payload.upperBound && state.cost + estimate > payload.upperBound) return;
    bestCost.set(state.exactSignature, state.cost);
    const topology = forward ? 0.2 * topologyPenalty(state.boxes, board) : 0;
    frontier.push([state.cost + estimate + topology, order++, state]);
  });

  while (frontier.length) {
    const current = frontier.pop()[2];
    if (bestCost.get(current.exactSignature) !== current.cost) continue;
    const reachable = reachablePaths(current, board);
    if (forward && createsSealedCorralDeadlock(current, board, reachable)) continue;
    const signature = pushKey(current, reachable);
    if (closed.has(signature)) continue;
    closed.add(signature); visited++;
    records.push({
      id: signature,
      parent: current.parent ?? null,
      segment: encodeMoves(current.segment || []),
      robot: current.robot,
    });

    if (records.length >= 500) flushRecords(records);
    if (payload.maxVisited && visited >= payload.maxVisited) {
      flushRecords(records);
      postMessage({type: "progress", visited, delta: visited - reported});
      postMessage({type: "done", visited, cutoff: true});
      return;
    }
    const nextStates = forward
      ? pushNeighbors(current, board, reachable).map(next => ({
          robot: next.robot,
          boxes: next.boxes,
          cost: current.cost + 1,
          parent: signature,
          segment: next.path,
        }))
      : reversePullNeighbors(current, board, reachable).map(next => ({
          ...next,
          parent: signature,
        }));
    for (const next of nextStates) {
      next.exactSignature = exactPushKey(next);
      if (next.cost >= (bestCost.get(next.exactSignature) ?? Infinity)) continue;
      const estimate = forward
        ? heuristic(next.boxes, board)
        : homeHeuristic(next.boxes, initialTargets);
      if (!Number.isFinite(estimate)) continue;
      if (payload.upperBound && next.cost + estimate > payload.upperBound) continue;
      bestCost.set(next.exactSignature, next.cost);
      const weightedEstimate = (forward ? 1.4 : 1.2) * estimate;
      const topology = forward ? 0.2 * topologyPenalty(next.boxes, board) : 0;
      frontier.push([next.cost + weightedEstimate + topology, order++, next]);
    }
    if (visited % 1000 === 0) {
      postMessage({type: "progress", visited, delta: visited - reported});
      reported = visited;
    }
  }
  flushRecords(records);
  postMessage({type: "progress", visited, delta: visited - reported});
  postMessage({type: "done", visited});
}

function search(payload) {
  if (payload.algorithm === "push-beam") return beamSearch(payload);
  if (payload.algorithm === "push-beam-restarts") return beamRestartSearch(payload);
  if (["ultimate", "portfolio", "fast"].includes(payload.algorithm)) {
    const beam = beamSearch({...payload, algorithm: "push-beam"});
    if (beam.path) return {...beam, strategy: "Push Beam"};
    const greedy = search({...payload, algorithm: "push-greedy"});
    if (greedy.path) return {...greedy, strategy: "Push Greedy"};
    const weighted = search({...payload, algorithm: "weighted-push-astar"});
    if (weighted.path) return {...weighted, strategy: "Weighted Push A*"};
    return search({...payload, algorithm: "push-astar"});
  }
  const board = parse(payload.state), initial = {
    robot: payload.state.robot,
    boxes: payload.state.boxes.map(([p, label]) => [...p.split(",").map(Number), label]),
    cost: 0,
    parent: null,
    segment: [],
  };
  const algorithm = payload.algorithm, frontier = new Heap(), seen = new Map(), cameFrom = new Map();
  const bestCost = new Map(), closed = new Set();
  const pushMacro = ["push-astar", "push-greedy", "weighted-push-astar"].includes(algorithm);
  const weight = algorithm === "weighted-push-astar" ? 1.6 : 1;
  let order = 0, visited = 0;
  const score = (s) => algorithm === "bfs" ? s.cost :
    algorithm === "dfs" ? -s.cost :
    ["greedy", "push-greedy"].includes(algorithm)
      ? heuristic(s.boxes, board) + 0.3 * topologyPenalty(s.boxes, board) :
    s.cost + weight * heuristic(s.boxes, board) +
      (algorithm === "weighted-push-astar" ? 0.15 * topologyPenalty(s.boxes, board) : 0);
  if (pushMacro) {
    initial.exactSignature = exactPushKey(initial);
    bestCost.set(initial.exactSignature, 0);
  }
  const initialScore = score(initial);
  if (!Number.isFinite(initialScore)) return {path: null, visited: 0};
  frontier.push([initialScore, order++, initial]);
  while (frontier.length) {
    const current = frontier.pop()[2];
    if (pushMacro && bestCost.get(current.exactSignature) !== current.cost) continue;
    const reachable = pushMacro ? reachablePaths(current, board) : null;
    const signature = pushMacro ? pushKey(current, reachable) : exactPushKey(current);
    if (pushMacro) {
      if (closed.has(signature)) continue;
      closed.add(signature);
    } else {
      if (seen.has(signature) && seen.get(signature) <= current.cost) continue;
      seen.set(signature, current.cost);
    }
    visited++;
    if (current.parent !== null) {
      cameFrom.set(signature, {parent: current.parent, segment: current.segment});
    }
    if (goal(current.boxes, board.goals)) return {path: reconstructPath(cameFrom, signature), visited};
    if (pushMacro && createsSealedCorralDeadlock(current, board, reachable)) continue;
    if (payload.maxVisited && visited >= payload.maxVisited) {
      return {path: null, visited, cutoff: true};
    }
    const nextStates = pushMacro ? pushNeighbors(current, board, reachable)
      .map(next => expandPushMacro(next, board, payload.forcedMacros !== false))
      .filter(Boolean) :
      neighbors(current, board).map(n => ({robot: n.robot, boxes: n.boxes, path: [n.move]}));
    for (const next of nextStates) {
      const child = {robot: next.robot, boxes: next.boxes,
        cost: current.cost + (pushMacro ? next.pushes : next.path.length),
        parent: signature, segment: next.path};
      if (pushMacro) {
        child.exactSignature = exactPushKey(child);
        if (child.cost >= (bestCost.get(child.exactSignature) ?? Infinity)) continue;
        const childScore = score(child);
        if (!Number.isFinite(childScore)) continue;
        if (payload.upperBound && child.cost + heuristic(child.boxes, board) > payload.upperBound) continue;
        bestCost.set(child.exactSignature, child.cost);
        frontier.push([childScore, order++, child]);
      } else {
        frontier.push([score(child), order++, child]);
      }
    }
    if (visited % 10000 === 0) postMessage({type: "progress", visited});
  }
  return {path: null, visited};
}
onmessage = ({data}) => {
  if (data.mode === "bidir-forward" || data.mode === "bidir-reverse") {
    bidirectionalSide(data);
  } else {
    postMessage({type: "done", ...search(data)});
  }
};
