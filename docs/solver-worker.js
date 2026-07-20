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
    this.evictions = 0;
  }
  get(key) { return this.values.get(key); }
  has(key) { return this.values.has(key); }
  set(key, value) {
    if (this.values.has(key)) this.values.delete(key);
    this.values.set(key, value);
    while (this.values.size > this.limit) {
      this.values.delete(this.values.keys().next().value);
      this.evictions++;
    }
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
  retainBest(limit) {
    if (this.items.length <= limit) return;
    const kept = this.items
      .sort((left, right) => left[0] - right[0] || left[1] - right[1])
      .slice(0, limit);
    this.items = [];
    kept.forEach(item => this.push(item));
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
      const approach = new Map(), approachQueue = [];
      for (const neighbor of floorNeighbors(gate, floor)) {
        if (cells.has(neighbor)) continue;
        approach.set(neighbor, 1);
        approachQueue.push(neighbor);
      }
      for (let head = 0; head < approachQueue.length; head++) {
        const position = approachQueue[head], distance = approach.get(position);
        if (distance >= 3) continue;
        for (const next of floorNeighbors(position, floor)) {
          if (next === gate || cells.has(next) || approach.has(next)) continue;
          approach.set(next, distance + 1);
          approachQueue.push(next);
        }
      }
      candidates.push({gate, cells, goals: roomGoals, depths, traffic, dependencies, approach});
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
    const currentLabels = new Map(), targetLabels = new Map();
    boxesInside.forEach(([, , label]) => currentLabels.set(label, (currentLabels.get(label) || 0) + 1));
    room.goals.forEach(goal => {
      const label = board.goals.get(goal);
      targetLabels.set(label, (targetLabels.get(label) || 0) + 1);
    });
    const labels = new Set([...currentLabels.keys(), ...targetLabels.keys()]);
    let labelFlow = 0;
    labels.forEach(label => {
      labelFlow += Math.abs((currentLabels.get(label) || 0) - (targetLabels.get(label) || 0));
    });
    penalty += 2 * labelFlow;
    for (const [y, x, label] of boxesInside) {
      const position = pkey(y, x);
      if (board.goals.get(position) !== label) penalty += 0.35 * (room.traffic.get(position) || 0);
    }

    const unsolved = room.goals.filter(goal => occupied.get(goal) !== board.goals.get(goal));
    const trafficDemand = unsolved.length + labelFlow;
    for (const [position, distance] of room.approach) {
      if (occupied.has(position)) penalty += 0.75 * trafficDemand * (4 - distance);
    }
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

function roomEvacuationPenalty(boxes, board) {
  const occupied = new Set(boxes.map(([y, x]) => pkey(y, x)));
  let penalty = 0;
  for (const room of board.topology.rooms) {
    const current = new Map(), target = new Map();
    const inside = boxes.filter(([y, x]) => room.cells.has(pkey(y, x)));
    inside.forEach(([, , label]) => current.set(label, (current.get(label) || 0) + 1));
    room.goals.forEach(goal => {
      const label = board.goals.get(goal);
      target.set(label, (target.get(label) || 0) + 1);
    });
    let surplus = 0;
    current.forEach((count, label) => {
      surplus += Math.max(0, count - (target.get(label) || 0));
    });
    if (!surplus) continue;
    penalty += 20 * inside.length + 8 * surplus;
    for (const [position, distance] of room.approach) {
      if (occupied.has(position)) penalty += 3 * (4 - distance);
    }
  }
  return penalty;
}

function roomFlowSignature(boxes, board) {
  return board.topology.rooms.map((room, index) => {
    const labels = boxes
      .filter(([y, x]) => room.cells.has(pkey(y, x)))
      .map(([, , label]) => label)
      .sort()
      .join("");
    return `${index}:${labels}`;
  }).join("|");
}

function roomTransitionEvent(before, after, board) {
  const previous = roomFlowSignature(before, board);
  const current = roomFlowSignature(after, board);
  return previous === current ? null : current;
}

function analyzePuzzleForSearch(data) {
  const board = parse(data);
  const boxes = data.boxes.map(([position, label]) => [
    ...position.split(",").map(Number), label,
  ]);
  const initial = {robot: data.robot, boxes};
  const labels = {};
  boxes.forEach(([, , label]) => { labels[label] = (labels[label] || 0) + 1; });
  const initialHeuristic = heuristic(boxes, board);
  const evacuationPenalty = roomEvacuationPenalty(boxes, board);
  const legalPushes = pushNeighbors(initial, board).length;
  const solvedBoxes = boxes.filter(([y, x, label]) =>
    board.goals.get(pkey(y, x)) === label).length;
  const roomSummaries = board.topology.rooms.map(room => {
    const inside = boxes.filter(([y, x]) => room.cells.has(pkey(y, x)));
    const current = {}, target = {};
    inside.forEach(([, , label]) => { current[label] = (current[label] || 0) + 1; });
    room.goals.forEach(goal => {
      const label = board.goals.get(goal);
      target[label] = (target[label] || 0) + 1;
    });
    const surplus = Object.entries(current).reduce((sum, [label, count]) =>
      sum + Math.max(0, count - (target[label] || 0)), 0);
    return {
      gate: room.gate,
      cells: room.cells.size,
      goals: room.goals.length,
      boxes: inside.length,
      surplus,
      dependencies: room.dependencies.length,
      maxDepth: Math.max(0, ...room.depths.values()),
    };
  });
  const searchScale = boxes.length * board.floor.size;
  const dependencyCount = roomSummaries.reduce((sum, room) => sum + room.dependencies, 0);
  const surplusBoxes = roomSummaries.reduce((sum, room) => sum + room.surplus, 0);
  const pressure = searchScale * (1 + 0.18 * board.topology.rooms.length) *
    (1 + 0.08 * dependencyCount) * (1 + 0.06 * Math.max(0, legalPushes - 2));
  const difficulty = pressure >= 1800 ? "extreme" : pressure >= 700 ? "complex" :
    pressure >= 180 ? "moderate" : "small";
  const phases = [];
  if (surplusBoxes) phases.push({id: "evacuation", reason: `${surplusBoxes} surplus room box${surplusBoxes === 1 ? "" : "es"}`});
  if (board.topology.rooms.length) phases.push({id: "room-packing", reason: `${board.topology.rooms.length} gated goal room${board.topology.rooms.length === 1 ? "" : "s"}`});
  if (board.topology.tunnels.size) phases.push({id: "tunnel-macros", reason: `${board.topology.tunnels.size} tunnel cells`});
  if (difficulty === "complex" || difficulty === "extreme") {
    phases.push({id: "milestone-reverse", reason: "large canonical push space"});
    phases.push({id: "landmark-bridges", reason: "connect forward phases to reverse layouts"});
  }
  phases.push({id: "exact-proof", reason: "complete fallback after heuristic workers"});
  const recommendations = {
    reverseWorkerLimit: difficulty === "extreme" ? 1 : difficulty === "complex" ? 2 : 3,
    sideVisitedLimit: difficulty === "extreme" ? 100000 : difficulty === "complex" ? 200000 : 250000,
    beamAttempts: difficulty === "small" ? 1 : 2,
    beamWidth: difficulty === "extreme" ? 300 : difficulty === "complex" ? 700 : 1200,
    beamVisited: difficulty === "extreme" ? 110000 : difficulty === "complex" ? 180000 : 250000,
    useEvacuation: surplusBoxes > 0,
    useSequenceMacros: board.topology.tunnels.size > 0 || board.topology.rooms.length > 0,
    useMilestoneReverse: difficulty === "complex" || difficulty === "extreme",
    checkpointLimit: difficulty === "extreme" ? 12 : 8,
  };
  return {
    dimensions: {rows: data.rows.length, columns: Math.max(...data.rows.map(row => row.length))},
    floorCells: board.floor.size,
    boxes: boxes.length,
    goals: board.goals.size,
    labels,
    solvedBoxes,
    initialHeuristic,
    legalPushes,
    articulations: board.topology.articulations.size,
    tunnelCells: board.topology.tunnels.size,
    rooms: roomSummaries,
    surplusBoxes,
    evacuationPenalty,
    dependencyCount,
    searchScale,
    pressure: Math.round(pressure),
    difficulty,
    phases,
    recommendations,
  };
}

function stratifiedCheckpoints(checkpoints) {
  const bands = new Map();
  for (const checkpoint of checkpoints) {
    if (!bands.has(checkpoint.checkpointBand)) bands.set(checkpoint.checkpointBand, []);
    bands.get(checkpoint.checkpointBand).push(checkpoint);
  }
  const queues = [...bands.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, items]) => items);
  const result = [];
  for (let depth = 0; queues.some(items => depth < items.length); depth++) {
    for (const items of queues) if (depth < items.length) result.push(items[depth]);
  }
  return result;
}
function targetMapFromBoxes(boxes, board) {
  const targets = new Map();
  boxes.forEach(([y, x, label]) => {
    if (!targets.has(label)) targets.set(label, []);
    targets.get(label).push({distances: playerAwarePushDistances(board, pkey(y, x))});
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

function targetLayoutHeuristic(boxes, targetBoxes, board, memo) {
  const signature = boxSignature(boxes);
  if (memo.has(signature)) return memo.get(signature);
  const targetsByLabel = new Map();
  targetBoxes.forEach(([y, x, label]) => {
    if (!targetsByLabel.has(label)) targetsByLabel.set(label, []);
    targetsByLabel.get(label).push(pkey(y, x));
  });
  const byLabel = new Map();
  boxes.forEach(([y, x, label]) => {
    if (!byLabel.has(label)) byLabel.set(label, []);
    byLabel.get(label).push([y, x]);
  });
  let total = 0;
  for (const [label, positions] of byLabel) {
    const targets = targetsByLabel.get(label) || [];
    const costs = positions.map(([y, x]) => {
      const distances = playerAwarePushDistances(board, pkey(y, x));
      return targets.map(target => distances.get(target) ?? Infinity);
    });
    total += minimumAssignmentCost(costs);
  }
  return memoizeBounded(memo, signature, total);
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
        pushedFrom: pkey(y, x),
        pushedTo: dest,
      });
    }
  });
  return result;
}

function pushBoxNeighbors(state, board, boxPosition, reachable = reachablePaths(state, board)) {
  const occupied = new Map(state.boxes.map((box, index) => [pkey(box[0], box[1]), index]));
  const index = occupied.get(boxPosition);
  if (index === undefined) return [];
  const [y, x, label] = state.boxes[index], result = [];
  for (const [move, [dy, dx]] of Object.entries(DIRS)) {
    const support = pkey(y - dy, x - dx), destination = pkey(y + dy, x + dx);
    if (!reachable.has(support) || !board.floor.has(destination) || occupied.has(destination)) continue;
    const boxes = state.boxes.map((box, boxIndex) =>
      boxIndex === index ? [y + dy, x + dx, label] : box);
    if (staticDead(y + dy, x + dx, board, label) ||
        createsDynamicDeadlock(boxes, board, [y + dy, x + dx])) continue;
    result.push({
      robot: [y, x],
      boxes,
      path: [...reachable.get(support), move],
      pushClass: `${label}:${y},${x}:${move}`,
      pushedFrom: boxPosition,
      pushedTo: destination,
    });
  }
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

function expandPushSequences(first, board, maxPushes = 12, maxExplored = 48, maxReturned = 8) {
  const initial = {...first, pushes: 1};
  const queue = [initial], endpoints = [];
  const seen = new Set([exactPushKey(initial)]);
  let head = 0;
  for (; head < queue.length && queue.length < maxExplored; head++) {
    const current = queue[head];
    if (current.pushes >= maxPushes || goal(current.boxes, board.goals)) {
      endpoints.push(current);
      continue;
    }
    const state = {robot: current.robot, boxes: current.boxes};
    const reachable = reachablePaths(state, board);
    if (createsSealedCorralDeadlock(state, board, reachable)) {
      endpoints.push(current);
      continue;
    }
    const continuations = pushBoxNeighbors(state, board, current.pushedTo, reachable);
    if (!continuations.length) endpoints.push({...current, macroDecision: true});
    else if (continuations.length > 1) endpoints.push({...current, macroDecision: true});
    for (const next of continuations) {
      const sequence = {
        robot: next.robot,
        boxes: next.boxes,
        path: [...current.path, ...next.path],
        pushes: current.pushes + 1,
        pushClass: `${first.pushClass}:${current.pushes + 1}`,
        pushedFrom: next.pushedFrom,
        pushedTo: next.pushedTo,
      };
      const signature = exactPushKey(sequence);
      if (seen.has(signature)) continue;
      seen.add(signature);
      queue.push(sequence);
      if (queue.length >= maxExplored) break;
    }
  }
  endpoints.push(...queue.slice(head));
  endpoints.sort((left, right) =>
    Number(Boolean(right.macroDecision)) - Number(Boolean(left.macroDecision)) ||
    right.pushes - left.pushes);
  const selected = [], destinations = new Set();
  for (const endpoint of endpoints) {
    if (destinations.has(endpoint.pushedTo)) continue;
    destinations.add(endpoint.pushedTo);
    selected.push(endpoint);
    if (selected.length >= maxReturned) break;
  }
  return [initial, ...selected.filter(endpoint => exactPushKey(endpoint) !== exactPushKey(initial))];
}

function expandStraightPushes(first, board, maxPushes = 8) {
  const [fromY, fromX] = first.pushedFrom.split(",").map(Number);
  const [toY, toX] = first.pushedTo.split(",").map(Number);
  const dy = toY - fromY, dx = toX - fromX;
  const results = [{...first, pushes: 1}];
  let current = results[0];
  while (current.pushes < maxPushes && !goal(current.boxes, board.goals)) {
    const state = {robot: current.robot, boxes: current.boxes};
    const reachable = reachablePaths(state, board);
    if (createsSealedCorralDeadlock(state, board, reachable)) break;
    const [y, x] = current.pushedTo.split(",").map(Number);
    const destination = pkey(y + dy, x + dx);
    const next = pushBoxNeighbors(state, board, current.pushedTo, reachable)
      .find(candidate => candidate.pushedTo === destination);
    if (!next) break;
    current = {
      ...next,
      path: [...current.path, ...next.path],
      pushes: current.pushes + 1,
      pushClass: `${first.pushClass}:${current.pushes + 1}`,
    };
    results.push(current);
  }
  return results;
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

function serializeSearchCheckpoint(candidate, board) {
  if (!candidate) return null;
  return {
    state: {
      rows: board.rows,
      robot: candidate.robot,
      boxes: candidate.boxes.map(([y, x, label]) => [pkey(y, x), label]),
    },
    path: reconstructNodePath(candidate.node),
    cost: candidate.cost,
    estimate: candidate.estimate,
  };
}

function takeDiverse(candidates, count, selected, scoreKey, groupKey = "pushClass") {
  const groups = new Map();
  for (const candidate of candidates) {
    if (selected.has(candidate.exactSignature)) continue;
    const key = candidate[groupKey] || candidate.pushClass || candidate.exactSignature;
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
  const ratios = profile === "milestone"
    ? [0.20, 0.20, 0.20, 0.40]
    : profile === "detour"
    ? [0.30, 0.25, 0.25, 0.20]
    : [0.50, 0.25, 0.15, 0.10];
  const groupKey = profile === "milestone" ? "strategicClass" : "pushClass";
  const selected = new Set(), result = [];
  bands.forEach((band, index) => {
    const quota = index === bands.length - 1
      ? width - ratios.slice(0, index).reduce((total, ratio) => total + Math.floor(width * ratio), 0)
      : Math.floor(width * ratios[index]);
    const scoreKey = index === bands.length - 1 ? "exploreScore" : "score";
    result.push(...takeDiverse(band, quota, selected, scoreKey, groupKey));
  });
  if (result.length < width) {
    const ranked = [...candidates].sort((left, right) => left.score - right.score);
    result.push(...takeDiverse(ranked, width - result.length, selected, "score", groupKey));
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
  const evacuationWeight = payload.evacuationWeight ?? 0;
  const beamProfile = payload.beamProfile || "balanced";
  const seed = payload.seed || 0;
  const transpositionLimit = payload.transpositionLimit || Math.max(12000, width * 60);
  const seenDepth = new BoundedDepthMap(transpositionLimit);
  const seenExactDepth = new BoundedDepthMap(Math.max(8000, Math.floor(transpositionLimit / 2)));
  const handoffLimit = payload.checkpointLimit || 12;
  const progressInterval = payload.progressInterval || 5000;
  const handoffCheckpoints = new Map();
  let visited = 0, reported = 0, bestEstimate = Infinity, bestPushes = 0;
  let beamCutoff = false;
  let bestCheckpoint = null;
  let bestHandoff = null;
  let phaseHandoff = null;
  const endgameCheckpoints = [];
  let trackedThrough = payload.trackedSignatures ? 0 : undefined;

  initial.reachable = reachablePaths(initial, board);
  if (createsSealedCorralDeadlock(initial, board, initial.reachable)) {
    return {path: null, visited};
  }
  initial.signature = pushKey(initial, initial.reachable);
  initial.strategicHistory = "";
  initial.openingHistory = "";
  const initialEstimate = heuristic(initial.boxes, board);
  if (!Number.isFinite(initialEstimate)) return {path: null, visited};
  bestEstimate = initialEstimate;
  seenDepth.set(initial.signature, 0);
  let beam = [initial];

  searchLayers: for (let depth = 0; beam.length && depth <= maxDepth; depth++) {
    const candidates = new Map();
    for (const current of beam) {
      visited++;
      if (goal(current.boxes, board.goals)) {
        return {path: reconstructNodePath(current.node), visited};
      }
      if (payload.maxVisited && visited >= payload.maxVisited) {
        beamCutoff = true;
        break searchLayers;
      }
      for (const rawNext of pushNeighbors(current, board, current.reachable)) {
        const expansions = payload.straightMacros
          ? expandStraightPushes(rawNext, board, payload.straightMacroLimit || 8)
          : payload.sequenceMacros
          ? expandPushSequences(
              rawNext,
              board,
              payload.sequenceMacroLimit || 12,
              payload.sequenceMacroExplored || 48,
              payload.sequenceMacroResults || 8,
            )
          : [expandPushMacro(rawNext, board, payload.forcedMacros !== false)].filter(Boolean);
        for (const next of expansions) {
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
        const evacuation = evacuationWeight
          ? roomEvacuationPenalty(child.boxes, board)
          : 0;
        if (beamProfile === "milestone") {
          const transition = roomTransitionEvent(current.boxes, child.boxes, board);
          child.strategicHistory = transition
            ? `${current.strategicHistory || ""}>${transition}`.split(">").slice(-4).join(">")
            : current.strategicHistory || "";
          child.openingHistory = child.cost <= 10
            ? `${current.openingHistory || ""}/${next.pushClass}`
            : current.openingHistory || "";
        }
        const score = (payload.costWeight || 0) * child.cost +
          weight * estimate + topologyWeight * topology +
          evacuationWeight * evacuation -
          goalPackingWeight * goalPackingBonus(child.boxes, board) +
          diversity * signatureNoise(child.exactSignature, seed);
        const exploreScore = topologyWeight * topology + evacuationWeight * evacuation -
          goalPackingWeight * goalPackingBonus(child.boxes, board) +
          diversity * signatureNoise(child.exactSignature, seed + 7919);
        const existing = candidates.get(child.exactSignature);
        if (!existing || score < existing.score) {
          const candidate = {
            ...child,
            node: {parent: current.node || null, segment: next.path},
            estimate,
            topology,
            evacuation,
            score,
            exploreScore,
            pushClass: next.pushClass,
            strategicClass: beamProfile === "milestone"
              ? `${child.openingHistory}|${child.strategicHistory}|${roomFlowSignature(child.boxes, board)}`
              : null,
            strategicHistory: child.strategicHistory,
            openingHistory: child.openingHistory,
          };
          candidates.set(child.exactSignature, candidate);
          if (!bestCheckpoint || estimate < bestCheckpoint.estimate ||
              (estimate === bestCheckpoint.estimate && child.cost < bestCheckpoint.cost)) {
            bestCheckpoint = candidate;
          }
          if ((payload.endgameVisited || payload.continuationVisited) &&
              estimate <= (payload.endgameThreshold || 60)) {
            const solvedGoals = candidate.boxes
              .filter(([y, x, label]) => board.goals.get(pkey(y, x)) === label)
              .map(([y, x, label]) => `${y},${x},${label}`)
              .sort()
              .join(";");
            candidate.checkpointClass =
              `${roomFlowSignature(candidate.boxes, board)}|${solvedGoals}|${next.pushClass}`;
            candidate.checkpointBand = Math.floor(estimate / 10);
            const existingCheckpoint = endgameCheckpoints.findIndex(checkpoint =>
              checkpoint.checkpointClass === candidate.checkpointClass);
            if (existingCheckpoint >= 0) {
              if (candidate.estimate >= endgameCheckpoints[existingCheckpoint].estimate) continue;
              endgameCheckpoints.splice(existingCheckpoint, 1);
            }
            endgameCheckpoints.push(candidate);
            endgameCheckpoints.sort((left, right) =>
              left.estimate - right.estimate ||
              (left.cost + left.estimate) - (right.cost + right.estimate) ||
              left.cost - right.cost);
            if (endgameCheckpoints.length > (payload.endgameCandidates || 24)) {
              const bandCounts = new Map();
              endgameCheckpoints.forEach(checkpoint => bandCounts.set(
                checkpoint.checkpointBand,
                (bandCounts.get(checkpoint.checkpointBand) || 0) + 1,
              ));
              let crowdedBand = null, crowdedCount = 0;
              for (const [band, count] of bandCounts) {
                if (count > crowdedCount || (count === crowdedCount && band < crowdedBand)) {
                  crowdedBand = band;
                  crowdedCount = count;
                }
              }
              for (let remove = endgameCheckpoints.length - 1; remove >= 0; remove--) {
                if (endgameCheckpoints[remove].checkpointBand !== crowdedBand) continue;
                endgameCheckpoints.splice(remove, 1);
                break;
              }
            }
          }
        }
        }
      }
      if (visited - reported >= progressInterval) {
        postMessage({type: "progress", visited: (payload.progressOffset || 0) + visited,
          bestEstimate, bestPushes, frontier: beam.length, depth});
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
      if (!bestHandoff || child.estimate < bestHandoff.estimate ||
          (child.estimate === bestHandoff.estimate && child.cost < bestHandoff.cost)) {
        bestHandoff = child;
      }
      if (!handoffCheckpoints.has(child.signature)) {
        handoffCheckpoints.set(child.signature, child);
        if (handoffCheckpoints.size > handoffLimit * 3) {
          const retained = [...handoffCheckpoints.entries()]
            .sort(([, left], [, right]) =>
              left.estimate - right.estimate ||
              (left.cost + left.estimate) - (right.cost + right.estimate))
            .slice(0, handoffLimit);
          handoffCheckpoints.clear();
          retained.forEach(([signature, checkpoint]) =>
            handoffCheckpoints.set(signature, checkpoint));
        }
      }
      if (evacuationWeight && child.evacuation === 0 &&
          (!phaseHandoff || child.cost + child.estimate <
            phaseHandoff.cost + phaseHandoff.estimate)) {
        phaseHandoff = child;
      }
    }
    beam = selectBeamLayer(beam, width, beamProfile);
    if (payload.trackedSignatures) {
      for (const child of beam) {
        if (payload.trackedSignatures[child.cost] === child.signature) {
          trackedThrough = Math.max(trackedThrough, child.cost);
        }
      }
    }
  }
  const probeCheckpoints = stratifiedCheckpoints(endgameCheckpoints);
  if (payload.continuationVisited && probeCheckpoints.length) {
    let remainingVisited = payload.continuationVisited;
    const profiles = payload.continuationProfiles?.length
      ? payload.continuationProfiles
      : [{beamProfile: "detour", weight: 3.5, topologyWeight: 0.6}];
    const attempts = Math.min(payload.continuationAttempts || 8, probeCheckpoints.length);
    for (let index = 0; index < attempts && remainingVisited > 0; index++) {
      const checkpoint = probeCheckpoints[index];
      const remainingBound = (payload.upperBound || maxDepth) - checkpoint.cost;
      const attemptVisited = Math.ceil(remainingVisited / (attempts - index));
      const continuation = beamSearch({
        ...payload,
        ...profiles[index % profiles.length],
        preparedBoard: undefined,
        state: {
          rows: board.rows,
          robot: checkpoint.robot,
          boxes: checkpoint.boxes.map(([y, x, label]) => [pkey(y, x), label]),
        },
        upperBound: remainingBound,
        maxDepth: remainingBound,
        maxVisited: attemptVisited,
        beamWidth: payload.continuationWidth || 36,
        transpositionLimit: payload.continuationTranspositionLimit || 10000,
        seed: seed + (index + 1) * 32452843,
        progressOffset: (payload.progressOffset || 0) + visited,
        continuationVisited: 0,
        endgameVisited: 0,
      });
      if (continuation.path) {
        return {
          path: [...reconstructNodePath(checkpoint.node), ...continuation.path],
          visited: visited + continuation.visited,
          bestEstimate: 0,
          bestPushes: checkpoint.cost,
          continuation: true,
        };
      }
      visited += continuation.visited;
      remainingVisited -= continuation.visited;
      if ((continuation.bestEstimate ?? Infinity) < bestEstimate) {
        bestEstimate = continuation.bestEstimate;
        bestPushes = checkpoint.cost + (continuation.bestPushes || 0);
      }
    }
  }
  if (payload.endgameVisited && probeCheckpoints.length) {
    let remainingVisited = payload.endgameVisited;
    const attempts = Math.min(payload.endgameAttempts || 12, probeCheckpoints.length);
    for (let index = 0; index < attempts && remainingVisited > 0; index++) {
      const checkpoint = probeCheckpoints[index];
      const remainingBound = (payload.upperBound || maxDepth) - checkpoint.cost;
      const attemptVisited = Math.ceil(remainingVisited / (attempts - index));
    const endgame = boundedPushDepthFirstSearch({
      algorithm: "bounded-push-dfs",
      state: {
        rows: board.rows,
        robot: checkpoint.robot,
        boxes: checkpoint.boxes.map(([y, x, label]) => [pkey(y, x), label]),
      },
      upperBound: remainingBound,
      maxDepth: remainingBound,
      maxVisited: attemptVisited,
      transpositionLimit: payload.endgameTranspositionLimit || 30000,
      dfsProfile: payload.endgameProfiles?.[index % payload.endgameProfiles.length] ||
        payload.endgameProfile || "balanced",
      diversity: payload.diversity,
      seed: seed + 15485863,
      progressOffset: (payload.progressOffset || 0) + visited,
      forcedMacros: false,
    });
    if (endgame.path) {
      return {
        path: [...reconstructNodePath(checkpoint.node), ...endgame.path],
        visited: visited + endgame.visited,
        bestEstimate: 0,
        bestPushes: checkpoint.cost,
        endgame: true,
      };
    }
    visited += endgame.visited;
      remainingVisited -= endgame.visited;
    }
  }
  return {
    path: null,
    visited,
    cutoff: beamCutoff,
    terminationReason: beamCutoff ? "budget" : "frontier-exhausted",
    bestEstimate,
    bestPushes,
    trackedThrough,
    checkpoint: serializeSearchCheckpoint(bestHandoff, board),
    checkpoints: [...handoffCheckpoints.values()]
      .sort((left, right) =>
        left.estimate - right.estimate ||
        (left.cost + left.estimate) - (right.cost + right.estimate))
      .slice(0, handoffLimit)
      .map(checkpoint => serializeSearchCheckpoint(checkpoint, board)),
    phaseCheckpoint: serializeSearchCheckpoint(phaseHandoff, board),
  };
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
  return {path: null, visited, cutoff: true, terminationReason: "restart-budget",
    bestEstimate, bestPushes, restarts: restartCount};
}

function boundedPushDepthFirstSearch(payload) {
  const board = parse(payload.state);
  const initial = {
    robot: payload.state.robot,
    boxes: payload.state.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]),
  };
  const bound = payload.upperBound ?? payload.pushBound ?? 300;
  const maxVisited = payload.maxVisited || 250000;
  const maxDepth = payload.maxDepth || bound;
  const seed = payload.seed || 0;
  const profile = payload.dfsProfile || "balanced";
  const discrepancyLimit = payload.discrepancyLimit ?? Infinity;
  const transpositions = new BoundedDepthMap(payload.transpositionLimit || 60000);
  const activePath = new Set(), segments = [];
  const checkpointLimit = payload.checkpointLimit || 8;
  const checkpoints = new Map();
  let visited = 0, reported = 0, cutoff = false, solution = null;
  let bestEstimate = Infinity, bestPushes = 0;
  let bestCheckpoint = null;
  let trackedThrough = payload.trackedSignatures ? 0 : undefined;

  const visit = (state, cost, discrepancyRemaining) => {
    if (cutoff || solution) return;
    visited++;
    if (visited >= maxVisited) {
      cutoff = true;
      return;
    }
    if (visited - reported >= 5000) {
      postMessage({type: "progress", visited: (payload.progressOffset || 0) + visited,
        bestEstimate, bestPushes, depth: cost, retained: transpositions.size});
      reported = visited;
    }
    if (goal(state.boxes, board.goals)) {
      solution = segments.flatMap(segment => segment);
      return;
    }
    const reachable = reachablePaths(state, board);
    if (createsSealedCorralDeadlock(state, board, reachable)) return;
    const signature = pushKey(state, reachable);
    if (payload.trackedSignatures?.[cost] === signature) {
      trackedThrough = Math.max(trackedThrough, cost);
    }
    if (activePath.has(signature) || (transpositions.get(signature) ?? Infinity) <= cost) return;
    activePath.add(signature);
    transpositions.set(signature, cost);

    const candidates = [];
    for (const rawNext of pushNeighbors(state, board, reachable)) {
      const next = expandPushMacro(rawNext, board, payload.forcedMacros !== false);
      if (!next) continue;
      const childCost = cost + next.pushes;
      if (childCost > maxDepth || childCost > bound) continue;
      const estimate = heuristic(next.boxes, board);
      if (!Number.isFinite(estimate) || childCost + estimate > bound) continue;
      const checkpointSignature = exactPushKey(next);
      if (!checkpoints.has(checkpointSignature)) {
        checkpoints.set(checkpointSignature, {
          state: {
            rows: board.rows,
            robot: next.robot,
            boxes: next.boxes.map(([y, x, label]) => [pkey(y, x), label]),
          },
          path: [...segments.flatMap(segment => segment), ...next.path],
          cost: childCost,
          estimate,
        });
        if (checkpoints.size > checkpointLimit * 3) {
          const retained = [...checkpoints.entries()]
            .sort(([, left], [, right]) =>
              left.estimate - right.estimate ||
              (left.cost + left.estimate) - (right.cost + right.estimate))
            .slice(0, checkpointLimit);
          checkpoints.clear();
          retained.forEach(([signature, checkpoint]) =>
            checkpoints.set(signature, checkpoint));
        }
      }
      if (estimate < bestEstimate) {
        bestEstimate = estimate;
        bestPushes = childCost;
        bestCheckpoint = {
          state: {
            rows: board.rows,
            robot: next.robot,
            boxes: next.boxes.map(([y, x, label]) => [pkey(y, x), label]),
          },
          path: [...segments.flatMap(segment => segment), ...next.path],
          cost: childCost,
          estimate,
        };
      }
      const topology = topologyPenalty(next.boxes, board);
      const evacuation = profile === "evacuation" ? roomEvacuationPenalty(next.boxes, board) : 0;
      const packing = goalPackingBonus(next.boxes, board);
      let score = 2.5 * estimate + topology - 0.8 * packing;
      if (profile === "detour") score = 1.5 * estimate + 1.4 * topology - packing;
      if (profile === "setup" && childCost <= 12) score = -estimate + topology - packing;
      if (profile === "room-flow") score = estimate + 6 * topology - packing;
      if (profile === "evacuation") score = estimate + 8 * evacuation + topology - packing;
      score += (payload.diversity ?? 1.5) * signatureNoise(exactPushKey(next), seed + childCost);
      candidates.push({next, cost: childCost, score});
    }
    candidates.sort((left, right) => left.score - right.score);
    for (let index = 0; index < candidates.length; index++) {
      const candidate = candidates[index];
      const discrepancy = index === 0 ? 0 : Math.ceil(Math.log2(index + 1));
      if (discrepancy > discrepancyRemaining) continue;
      segments.push(candidate.next.path);
      visit(
        {robot: candidate.next.robot, boxes: candidate.next.boxes},
        candidate.cost,
        discrepancyRemaining - discrepancy,
      );
      segments.pop();
      if (cutoff || solution) break;
    }
    activePath.delete(signature);
  };

  const initialEstimate = heuristic(initial.boxes, board);
  bestEstimate = initialEstimate;
  if (Number.isFinite(initialEstimate) && initialEstimate <= bound) {
    visit(initial, 0, discrepancyLimit);
  }
  return {
    path: solution,
    visited,
    cutoff,
    terminationReason: solution ? "solution" : cutoff ? "budget" : "profile-exhausted",
    bestEstimate,
    bestPushes,
    bound,
    discrepancyLimit,
    retained: transpositions.size,
    trackedThrough,
    checkpoint: bestCheckpoint,
    checkpoints: [...checkpoints.values()]
      .sort((left, right) =>
        left.estimate - right.estimate ||
        (left.cost + left.estimate) - (right.cost + right.estimate))
      .slice(0, checkpointLimit),
  };
}

function pushIterativeDeepeningAStar(payload) {
  const board = parse(payload.state);
  const initial = {
    robot: payload.state.robot,
    boxes: payload.state.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]),
  };
  const upperBound = payload.upperBound ?? payload.pushBound ?? 300;
  const maxVisited = payload.maxVisited || 300000;
  const seed = payload.seed || 0;
  const profile = payload.idaProfile || "balanced";
  const segments = [], activePath = new Set();
  let visited = 0, reported = 0, solution = null, cutoff = false;
  let bestEstimate = heuristic(initial.boxes, board), bestPushes = 0;
  let generated = 0, thresholdPrunes = 0, upperBoundPrunes = 0;
  let corralPrunes = 0, cyclePrunes = 0, transpositionPrunes = 0;
  let shardRejections = 0, shardAcceptances = 0, maxDepth = 0;
  let transpositionEvictions = 0, maxTranspositions = 0;
  let nextThresholdCandidate = Infinity;
  const progressInterval = payload.progressInterval || 25000;
  let trackedThrough = payload.trackedSignatures ? 0 : undefined;
  const exactShard = payload.exactShard;
  if (!Number.isFinite(bestEstimate)) {
    return {path: null, visited, cutoff: false, terminationReason: "infeasible-root",
      bestEstimate, bestPushes};
  }

  const orderingScore = (state, cost) => {
    const estimate = heuristic(state.boxes, board);
    const topology = topologyPenalty(state.boxes, board);
    const packing = goalPackingBonus(state.boxes, board);
    if (profile === "milestone") {
      return estimate + 2.5 * topology - packing +
        0.35 * signatureNoise(roomFlowSignature(state.boxes, board), seed + cost);
    }
    if (profile === "detour") return estimate + 1.25 * topology - packing;
    return estimate + 0.6 * topology - 0.8 * packing;
  };

  const searchContour = (state, cost, threshold, transpositions, shardAccepted = false) => {
    if (cutoff || solution) return Infinity;
    maxDepth = Math.max(maxDepth, cost);
    const estimate = heuristic(state.boxes, board);
    const total = cost + estimate;
    let accepted = shardAccepted;
    if (!accepted && exactShard && cost >= exactShard.depth) {
      const bucket = Math.floor(signatureNoise(exactPushKey(state), 0) * exactShard.count);
      if (bucket !== exactShard.index) {
        shardRejections++;
        return Infinity;
      }
      accepted = true;
      shardAcceptances++;
    }
    if (total > threshold) {
      thresholdPrunes++;
      nextThresholdCandidate = Math.min(nextThresholdCandidate, total);
      return total;
    }
    if (goal(state.boxes, board.goals)) {
      solution = segments.flatMap(segment => segment);
      return total;
    }
    if (++visited >= maxVisited) {
      cutoff = true;
      return Infinity;
    }
    if (visited - reported >= progressInterval) {
      postMessage({type: "progress", visited, threshold, bestEstimate, bestPushes,
        depth: cost, maxDepth, generated, thresholdPrunes, upperBoundPrunes,
        corralPrunes, cyclePrunes, transpositionPrunes,
        shardRejected: shardRejections, shardAccepted: shardAcceptances,
        transpositions: transpositions.size, transpositionEvictions: transpositions.evictions,
        nextThreshold: Number.isFinite(nextThresholdCandidate) ? nextThresholdCandidate : undefined});
      reported = visited;
    }
    const reachable = reachablePaths(state, board);
    if (createsSealedCorralDeadlock(state, board, reachable)) {
      corralPrunes++;
      return Infinity;
    }
    const signature = pushKey(state, reachable);
    if (payload.trackedSignatures?.[cost] === signature) {
      trackedThrough = Math.max(trackedThrough, cost);
    }
    if (activePath.has(signature)) {
      cyclePrunes++;
      return Infinity;
    }
    if ((transpositions.get(signature) ?? Infinity) <= cost) {
      transpositionPrunes++;
      return Infinity;
    }
    activePath.add(signature);
    transpositions.set(signature, cost);

    let nextThreshold = Infinity;
    const candidates = [];
    for (const rawNext of pushNeighbors(state, board, reachable)) {
      generated++;
      const next = expandPushMacro(rawNext, board, payload.forcedMacros !== false);
      if (!next) continue;
      const childCost = cost + next.pushes;
      if (childCost > upperBound) {
        upperBoundPrunes++;
        continue;
      }
      const childEstimate = heuristic(next.boxes, board);
      if (!Number.isFinite(childEstimate) || childCost + childEstimate > upperBound) {
        upperBoundPrunes++;
        continue;
      }
      if (childEstimate < bestEstimate) {
        bestEstimate = childEstimate;
        bestPushes = childCost;
      }
      candidates.push({
        next,
        cost: childCost,
        total: childCost + childEstimate,
        score: orderingScore(next, childCost) +
          (payload.diversity ?? 0.2) * signatureNoise(exactPushKey(next), seed + childCost),
      });
    }
    candidates.sort((left, right) => left.total - right.total || left.score - right.score);
    for (const candidate of candidates) {
      if (candidate.total > threshold) {
        thresholdPrunes++;
        nextThresholdCandidate = Math.min(nextThresholdCandidate, candidate.total);
        nextThreshold = Math.min(nextThreshold, candidate.total);
        continue;
      }
      segments.push(candidate.next.path);
      const exceeded = searchContour(
        {robot: candidate.next.robot, boxes: candidate.next.boxes},
        candidate.cost,
        threshold,
        transpositions,
        accepted,
      );
      segments.pop();
      nextThreshold = Math.min(nextThreshold, exceeded);
      if (cutoff || solution) break;
    }
    activePath.delete(signature);
    return nextThreshold;
  };

  let threshold = bestEstimate;
  while (Number.isFinite(threshold) && threshold <= upperBound && !cutoff && !solution) {
    postMessage({type: "contour", threshold, visited, exactShard});
    activePath.clear();
    const transpositions = new BoundedDepthMap(payload.transpositionLimit || 80000);
    nextThresholdCandidate = Infinity;
    const nextThreshold = searchContour(initial, 0, threshold, transpositions);
    transpositionEvictions += transpositions.evictions;
    maxTranspositions = Math.max(maxTranspositions, transpositions.size);
    if (!Number.isFinite(nextThreshold)) break;
    if (nextThreshold <= threshold) threshold++;
    else threshold = nextThreshold;
  }
  return {
    path: solution,
    visited,
    cutoff,
    terminationReason: solution ? "solution" : cutoff ? "budget" : "bound-exhausted",
    bestEstimate,
    bestPushes,
    threshold,
    bound: upperBound,
    trackedThrough,
    exactShard,
    generated,
    thresholdPrunes,
    upperBoundPrunes,
    corralPrunes,
    cyclePrunes,
    transpositionPrunes,
    shardRejected: shardRejections,
    shardAccepted: shardAcceptances,
    transpositionEvictions,
    maxTranspositions,
    maxDepth,
    nextThreshold: Number.isFinite(nextThresholdCandidate) ? nextThresholdCandidate : undefined,
  };
}

function bridgeAStarSearch(payload) {
  const board = parse(payload.state);
  const initial = {
    robot: payload.state.robot,
    boxes: payload.state.boxes.map(([position, label]) => [
      ...position.split(",").map(Number), label,
    ]),
    cost: 0,
  };
  const targetBoxes = payload.targetState.boxes.map(([position, label]) => [
    ...position.split(",").map(Number), label,
  ]);
  const targetState = {robot: payload.targetState.robot, boxes: targetBoxes};
  const targetReachable = reachablePaths(targetState, board);
  const targetKey = payload.targetId || pushKey(targetState, targetReachable);
  const heuristicMemo = new Map();
  const frontier = new Heap(), bestCost = new Map(), closed = new Set();
  let cameFrom = new Map();
  const weight = payload.weight || 1.4;
  const maxVisited = payload.maxVisited || 100000;
  const frontierLimit = payload.frontierLimit || 4000;
  let visited = 0, order = 0, bestEstimate = Infinity, bestCheckpoint = null;
  let compactions = 0, peakFrontier = 0;

  initial.reachable = reachablePaths(initial, board);
  initial.signature = pushKey(initial, initial.reachable);
  initial.estimate = targetLayoutHeuristic(initial.boxes, targetBoxes, board, heuristicMemo);
  const initialEstimate = initial.estimate;
  if (!Number.isFinite(initial.estimate)) {
    return {path: null, visited, cutoff: false,
      terminationReason: "target-incompatible", bestEstimate: initial.estimate};
  }
  delete initial.reachable;
  bestCost.set(initial.signature, 0);
  frontier.push([weight * initial.estimate, order++, initial]);

  while (frontier.length) {
    const current = frontier.pop()[2];
    if (bestCost.get(current.signature) !== current.cost || closed.has(current.signature)) continue;
    bestCost.delete(current.signature);
    closed.add(current.signature);
    visited++;
    if (current.signature === targetKey) {
      return {
        path: reconstructPath(cameFrom, current.signature),
        visited,
        terminationReason: "target-reached",
        initialEstimate,
        bestEstimate: 0,
        bestPushes: current.cost,
        peakFrontier,
        compactions,
        finalState: {
          rows: board.rows,
          robot: current.robot,
          boxes: current.boxes.map(([y, x, label]) => [pkey(y, x), label]),
        },
      };
    }
    if (visited >= maxVisited) break;
    const currentReachable = reachablePaths(current, board);
    for (const rawNext of pushNeighbors(current, board, currentReachable)) {
      const next = expandPushMacro(rawNext, board, payload.forcedMacros !== false);
      if (!next) continue;
      const child = {
        robot: next.robot,
        boxes: next.boxes,
        cost: current.cost + next.pushes,
      };
      if (payload.upperBound && child.cost > payload.upperBound) continue;
      child.reachable = reachablePaths(child, board);
      child.signature = pushKey(child, child.reachable);
      delete child.reachable;
      if (closed.has(child.signature) ||
          child.cost >= (bestCost.get(child.signature) ?? Infinity)) continue;
      child.estimate = targetLayoutHeuristic(child.boxes, targetBoxes, board, heuristicMemo);
      if (!Number.isFinite(child.estimate) ||
          (payload.upperBound && child.cost + child.estimate > payload.upperBound)) continue;
      if (child.estimate < bestEstimate) {
        bestEstimate = child.estimate;
        bestCheckpoint = {
          state: {
            rows: board.rows,
            robot: child.robot,
            boxes: child.boxes.map(([y, x, label]) => [pkey(y, x), label]),
          },
          cost: child.cost,
          estimate: child.estimate,
          signature: child.signature,
        };
      }
      bestCost.set(child.signature, child.cost);
      cameFrom.set(child.signature, {parent: current.signature, segment: next.path});
      frontier.push([child.cost + weight * child.estimate, order++, child]);
    }
    peakFrontier = Math.max(peakFrontier, frontier.length);
    if (frontier.length > frontierLimit * 2) {
      frontier.retainBest(frontierLimit);
      const retainedCosts = new Map();
      const ancestry = new Set([initial.signature]);
      const pending = [];
      for (const [, , state] of frontier.items) {
        const previous = retainedCosts.get(state.signature) ?? Infinity;
        if (state.cost < previous) retainedCosts.set(state.signature, state.cost);
        pending.push(state.signature);
      }
      if (bestCheckpoint?.signature) pending.push(bestCheckpoint.signature);
      while (pending.length) {
        const signature = pending.pop();
        if (ancestry.has(signature)) continue;
        ancestry.add(signature);
        const record = cameFrom.get(signature);
        if (record?.parent) pending.push(record.parent);
      }
      cameFrom = new Map([...cameFrom].filter(([signature]) => ancestry.has(signature)));
      bestCost.clear();
      retainedCosts.forEach((cost, signature) => bestCost.set(signature, cost));
      compactions++;
    }
    if (visited % 5000 === 0) postMessage({type: "progress", visited,
      bestEstimate, bestPushes: bestCheckpoint?.cost, frontier: frontier.length,
      retained: bestCost.size, peakFrontier, compactions});
  }
  const cutoff = visited >= maxVisited;
  const checkpoint = bestCheckpoint && {
    state: bestCheckpoint.state,
    path: reconstructPath(cameFrom, bestCheckpoint.signature),
    cost: bestCheckpoint.cost,
    estimate: bestCheckpoint.estimate,
  };
  return {path: null, visited, cutoff,
    terminationReason: cutoff ? "budget" : "frontier-exhausted",
    initialEstimate, bestEstimate, bestPushes: bestCheckpoint?.cost,
    frontier: frontier.length, retained: bestCost.size, peakFrontier, compactions,
    checkpoint};
}

function bidirectionalSide(payload) {
  const board = parse(payload.state);
  const initialBoxes = payload.state.boxes.map(([p, label]) => [...p.split(",").map(Number), label]);
  const initialTargets = targetMapFromBoxes(initialBoxes, board);
  const forward = payload.mode === "bidir-forward";
  const frontier = new Heap(), closed = new Set(), records = [];
  let bestCost = new Map();
  const frontierLimit = payload.frontierLimit || 40000;
  let order = 0, visited = 0, reported = 0, bestLandmarkEstimate = Infinity;
  let generated = 0, peakFrontier = 0, compactions = 0;
  const compactFrontier = () => {
    peakFrontier = Math.max(peakFrontier, frontier.length);
    if (frontier.length <= frontierLimit * 2) return;
    frontier.retainBest(frontierLimit);
    const retainedCosts = new Map();
    for (const [, , state] of frontier.items) {
      const previous = retainedCosts.get(state.exactSignature) ?? Infinity;
      if (state.cost < previous) retainedCosts.set(state.exactSignature, state.cost);
    }
    bestCost = retainedCosts;
    compactions++;
  };
  const landmarkCandidates = new Map();
  const emitLandmarks = () => {
    if (forward || !landmarkCandidates.size) return;
    const landmarks = stratifiedCheckpoints([...landmarkCandidates.values()])
      .slice(0, payload.landmarkLimit || 64)
      .map(({checkpointBand: _band, checkpointClass: _class, ...landmark}) => landmark);
    postMessage({type: "landmarks", landmarks});
  };
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
  compactFrontier();

  while (frontier.length) {
    const current = frontier.pop()[2];
    if (bestCost.get(current.exactSignature) !== current.cost) continue;
    bestCost.delete(current.exactSignature);
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
    const landmarkEstimate = forward
      ? heuristic(current.boxes, board)
      : homeHeuristic(current.boxes, initialTargets);
    if (landmarkEstimate < bestLandmarkEstimate) {
      bestLandmarkEstimate = landmarkEstimate;
      postMessage({
        type: "landmark",
        id: signature,
        estimate: landmarkEstimate,
        cost: current.cost,
        state: {
          rows: board.rows,
          robot: current.robot,
          boxes: current.boxes.map(([y, x, label]) => [pkey(y, x), label]),
        },
      });
    }
    if (!forward) {
      const solvedGoals = current.boxes
        .filter(([y, x, label]) => board.goals.get(pkey(y, x)) === label)
        .map(([y, x, label]) => `${y},${x},${label}`)
        .sort()
        .join(";");
      const checkpointBand = Math.floor(landmarkEstimate / 10);
      const checkpointClass =
        `${checkpointBand}|${roomFlowSignature(current.boxes, board)}|${solvedGoals}`;
      const existing = landmarkCandidates.get(checkpointClass);
      if (!existing || landmarkEstimate < existing.estimate ||
          (landmarkEstimate === existing.estimate && current.cost < existing.cost)) {
        landmarkCandidates.set(checkpointClass, {
          id: signature,
          estimate: landmarkEstimate,
          cost: current.cost,
          checkpointBand,
          checkpointClass,
          state: {
            rows: board.rows,
            robot: current.robot,
            boxes: current.boxes.map(([y, x, label]) => [pkey(y, x), label]),
          },
        });
      }
    }

    if (records.length >= 500) flushRecords(records);
    if (payload.maxVisited && visited >= payload.maxVisited) {
      flushRecords(records);
      emitLandmarks();
      postMessage({type: "progress", visited, delta: visited - reported,
        bestEstimate: bestLandmarkEstimate, frontier: frontier.length,
        retained: bestCost.size, generated, peakFrontier, compactions});
      postMessage({type: "done", visited, cutoff: true, terminationReason: "budget",
        bestEstimate: bestLandmarkEstimate, generated, peakFrontier, compactions,
        frontier: frontier.length, retained: bestCost.size});
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
      generated++;
      const weightedEstimate = (forward ? 1.4 : 1.2) * estimate;
      const topology = forward ? 0.2 * topologyPenalty(next.boxes, board) : 0;
      frontier.push([next.cost + weightedEstimate + topology, order++, next]);
    }
    compactFrontier();
    if (visited % 1000 === 0) {
      postMessage({type: "progress", visited, delta: visited - reported,
        bestEstimate: bestLandmarkEstimate, frontier: frontier.length,
        retained: bestCost.size, generated, peakFrontier, compactions});
      reported = visited;
    }
  }
  flushRecords(records);
  emitLandmarks();
  postMessage({type: "progress", visited, delta: visited - reported,
    bestEstimate: bestLandmarkEstimate, frontier: frontier.length,
    retained: bestCost.size, generated, peakFrontier, compactions});
  postMessage({type: "done", visited, cutoff: false, terminationReason: "exhausted",
    bestEstimate: bestLandmarkEstimate, generated, peakFrontier, compactions,
    frontier: frontier.length, retained: bestCost.size});
}

function search(payload) {
  if (payload.algorithm === "analyze-puzzle") {
    return {path: null, visited: 0, analysis: analyzePuzzleForSearch(payload.state)};
  }
  if (payload.algorithm === "bridge-astar") return bridgeAStarSearch(payload);
  if (payload.algorithm === "push-beam") return beamSearch(payload);
  if (payload.algorithm === "push-beam-restarts") return beamRestartSearch(payload);
  if (payload.algorithm === "bounded-push-dfs") return boundedPushDepthFirstSearch(payload);
  if (payload.algorithm === "push-ida-star") return pushIterativeDeepeningAStar(payload);
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
