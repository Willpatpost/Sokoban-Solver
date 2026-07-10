const DIRS = {Up: [-1, 0], Down: [1, 0], Left: [0, -1], Right: [0, 1]};
const OPPOSITE = {Up: "Down", Down: "Up", Left: "Right", Right: "Left"};
const key = (robot, boxes) => `${robot.join(",")}|${[...boxes].sort().join(";")}`;
const pkey = (y, x) => `${y},${x}`;
function hashKey(text) {
  let h1 = 0xdeadbeef ^ text.length, h2 = 0x41c6ce57 ^ text.length;
  for (let i = 0, ch; i < text.length; i++) {
    ch = text.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return 4294967296 * (2097151 & h2) + (h1 >>> 0);
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
function parse(data) {
  const floor = new Set(), walls = new Set(), goals = new Map();
  data.rows.forEach((row, y) => [...row].forEach((ch, x) => {
    const p = pkey(y, x);
    if (ch === "O") walls.add(p); else floor.add(p);
    if (ch === "S") goals.set(p, "X"); else if (/[a-z]/.test(ch)) goals.set(p, ch.toUpperCase());
  }));
  const pushDistances = new Map([...goals.keys()].map(goal => [goal, reversePushDistances(floor, goal)]));
  return {rows: data.rows, floor, walls, goals, pushDistances};
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
function heuristic(boxes, board) {
  const byLabel = new Map();
  boxes.forEach(([y, x, label]) => {
    if (!byLabel.has(label)) byLabel.set(label, []);
    byLabel.get(label).push([y, x]);
  });
  let total = 0;
  for (const [label, positions] of byLabel) {
    const targets = [...board.goals].filter(([, l]) => l === label).map(([p]) => p);
    const memo = new Map();
    const assign = (i, mask) => {
      if (i === positions.length) return 0;
      const mk = `${i},${mask}`; if (memo.has(mk)) return memo.get(mk);
      let best = Infinity;
      targets.forEach((target, j) => {
        if (!(mask & (1 << j))) {
          const [y, x] = positions[i];
          const distance = board.pushDistances.get(target)?.get(pkey(y, x)) ?? Infinity;
          best = Math.min(best, distance + assign(i + 1, mask | (1 << j)));
        }
      });
      memo.set(mk, best); return best;
    };
    total += assign(0, 0);
  }
  return total;
}
function goal(boxes, goals) {
  return boxes.every(([y, x, label]) => goals.get(pkey(y, x)) === label);
}
function corner(y, x, board, label) {
  if (board.goals.get(pkey(y, x)) === label) return false;
  const wall = (dy, dx) => board.walls.has(pkey(y + dy, x + dx));
  return (wall(-1, 0) && wall(0, -1)) || (wall(-1, 0) && wall(0, 1)) ||
    (wall(1, 0) && wall(0, -1)) || (wall(1, 0) && wall(0, 1));
}
function staticDead(y, x, board, label) {
  if (board.goals.get(pkey(y, x)) === label) return false;
  return ![...board.goals]
    .filter(([, goalLabel]) => goalLabel === label)
    .some(([goal]) => board.pushDistances.get(goal)?.has(pkey(y, x)));
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
      if (corner(by, bx, board, label)) continue;
      boxes = boxes.map((b, i) => i === index ? [by, bx, label] : b);
    }
    result.push({robot: [ny, nx], boxes, move});
  }
  return result;
}

function reachablePaths(state, board) {
  const occupied = new Set(state.boxes.map(b => pkey(b[0], b[1])));
  const start = pkey(state.robot[0], state.robot[1]);
  const paths = new Map([[start, []]]);
  const queue = [state.robot];
  for (let head = 0; head < queue.length; head++) {
    const [y, x] = queue[head], path = paths.get(pkey(y, x));
    for (const [move, [dy, dx]] of Object.entries(DIRS)) {
      const ny = y + dy, nx = x + dx, next = pkey(ny, nx);
      if (paths.has(next) || !board.floor.has(next) || occupied.has(next)) continue;
      paths.set(next, [...path, move]);
      queue.push([ny, nx]);
    }
  }
  return paths;
}

function pushNeighbors(state, board, reachable = reachablePaths(state, board)) {
  const occupied = new Map(state.boxes.map((b, i) => [pkey(b[0], b[1]), i]));
  const result = [];
  state.boxes.forEach(([y, x, label], index) => {
    for (const [move, [dy, dx]] of Object.entries(DIRS)) {
      const support = pkey(y - dy, x - dx), dest = pkey(y + dy, x + dx);
      if (!reachable.has(support) || !board.floor.has(dest) || occupied.has(dest)) continue;
      if (corner(y + dy, x + dx, board, label)) continue;
      const boxes = state.boxes.map((b, i) => i === index ? [y + dy, x + dx, label] : b);
      result.push({robot: [y, x], boxes, path: [...reachable.get(support), move]});
    }
  });
  return result;
}

function pushKey(state, reachable) {
  const robotRegion = [...reachable.keys()].sort()[0];
  return `${robotRegion}|${[...state.boxes.map(b => b.join(","))].sort().join(";")}`;
}

function invertWalk(path) {
  return [...path].reverse().map(move => OPPOSITE[move]);
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
      const walkToPullSpot = reachable.get(boxBefore);
      const walkFromPushLanding = invertWalk(walkToPullSpot);
      result.push({
        robot: [y - 2 * dy, x - 2 * dx],
        boxes,
        cost: state.cost + 1 + walkFromPushLanding.length,
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

function bidirectionalSide(payload) {
  const board = parse(payload.state);
  const initialBoxes = payload.state.boxes.map(([p, label]) => [...p.split(",").map(Number), label]);
  const forward = payload.mode === "bidir-forward";
  const frontier = new Heap(), seen = new Map(), records = [];
  let order = 0, visited = 0, reported = 0;
  const starts = forward
    ? [{robot: payload.state.robot, boxes: initialBoxes, cost: 0, path: []}]
    : reverseStartStates(board, initialBoxes, payload.reverseShard || {index: 0, count: 1});
  starts.forEach(state => frontier.push([state.cost + heuristic(state.boxes, board), order++, state]));

  while (frontier.length) {
    const current = frontier.pop()[2];
    const reachable = reachablePaths(current, board);
    const signature = pushKey(current, reachable);
    const hashed = hashKey(signature);
    if (seen.has(signature) && seen.get(signature) <= current.cost) continue;
    seen.set(signature, current.cost); visited++;
    records.push({
      h: hashed,
      key: signature,
      parent: current.parent ?? null,
      segment: current.segment || [],
      robot: current.robot,
    });

    if (records.length >= 500) flushRecords(records);
    const nextStates = forward
      ? pushNeighbors(current, board, reachable).map(next => ({
          robot: next.robot,
          boxes: next.boxes,
          cost: current.cost + next.path.length,
          parent: hashed,
          segment: next.path,
        }))
      : reversePullNeighbors(current, board, reachable).map(next => ({
          ...next,
          parent: hashed,
        }));
    for (const next of nextStates) {
      const score = forward
        ? next.cost + 1.4 * heuristic(next.boxes, board)
        : next.cost;
      frontier.push([score, order++, next]);
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
  if (["ultimate", "portfolio", "fast"].includes(payload.algorithm)) {
    const greedy = search({...payload, algorithm: "push-greedy"});
    if (greedy.path) return {...greedy, strategy: "Push Greedy"};
    const weighted = search({...payload, algorithm: "weighted-push-astar"});
    if (weighted.path) return {...weighted, strategy: "Weighted Push A*"};
    return search({...payload, algorithm: "push-astar"});
  }
  const board = parse(payload.state), initial = {
    robot: payload.state.robot,
    boxes: payload.state.boxes.map(([p, label]) => [...p.split(",").map(Number), label]),
    cost: 0, path: [],
  };
  const algorithm = payload.algorithm, frontier = new Heap(), seen = new Map();
  const pushMacro = ["push-astar", "push-greedy", "weighted-push-astar"].includes(algorithm);
  const weight = algorithm === "weighted-push-astar" ? 1.6 : 1;
  let order = 0, visited = 0;
  const score = (s) => algorithm === "bfs" ? s.cost :
    algorithm === "dfs" ? -s.cost :
    ["greedy", "push-greedy"].includes(algorithm) ? heuristic(s.boxes, board) :
    s.cost + weight * heuristic(s.boxes, board);
  frontier.push([score(initial), order++, initial]);
  while (frontier.length) {
    const current = frontier.pop()[2];
    const reachable = pushMacro ? reachablePaths(current, board) : null;
    const signature = pushMacro ? pushKey(current, reachable) :
      key(current.robot, current.boxes.map(b => b.join(",")));
    if (seen.has(signature) && seen.get(signature) <= current.cost) continue;
    seen.set(signature, current.cost); visited++;
    if (goal(current.boxes, board.goals)) return {path: current.path, visited};
    const nextStates = pushMacro ? pushNeighbors(current, board, reachable) :
      neighbors(current, board).map(n => ({robot: n.robot, boxes: n.boxes, path: [n.move]}));
    for (const next of nextStates) {
      const child = {robot: next.robot, boxes: next.boxes, cost: current.cost + next.path.length,
        path: [...current.path, ...next.path]};
      frontier.push([score(child), order++, child]);
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
