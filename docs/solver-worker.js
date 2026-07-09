const DIRS = {Up: [-1, 0], Down: [1, 0], Left: [0, -1], Right: [0, 1]};
const key = (robot, boxes) => `${robot.join(",")}|${[...boxes].sort().join(";")}`;
const pkey = (y, x) => `${y},${x}`;

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
  return {rows: data.rows, floor, walls, goals};
}
function heuristic(boxes, goals) {
  const byLabel = new Map();
  boxes.forEach(([y, x, label]) => {
    if (!byLabel.has(label)) byLabel.set(label, []);
    byLabel.get(label).push([y, x]);
  });
  let total = 0;
  for (const [label, positions] of byLabel) {
    const targets = [...goals].filter(([, l]) => l === label).map(([p]) => p.split(",").map(Number));
    const memo = new Map();
    const assign = (i, mask) => {
      if (i === positions.length) return 0;
      const mk = `${i},${mask}`; if (memo.has(mk)) return memo.get(mk);
      let best = Infinity;
      targets.forEach(([gy, gx], j) => {
        if (!(mask & (1 << j))) {
          const [y, x] = positions[i];
          best = Math.min(best, Math.abs(y - gy) + Math.abs(x - gx) + assign(i + 1, mask | (1 << j)));
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

function pushNeighbors(state, board) {
  const occupied = new Map(state.boxes.map((b, i) => [pkey(b[0], b[1]), i]));
  const reachable = reachablePaths(state, board), result = [];
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

function pushKey(state, board) {
  const robotRegion = [...reachablePaths(state, board).keys()].sort()[0];
  return `${robotRegion}|${[...state.boxes.map(b => b.join(","))].sort().join(";")}`;
}

function search(payload) {
  if (payload.algorithm === "portfolio" || payload.algorithm === "fast") {
    const greedy = search({...payload, algorithm: "push-greedy", maxVisited: 50000});
    if (greedy.path) return {...greedy, strategy: "Push Greedy"};
    const weighted = search({...payload, algorithm: "weighted-push-astar", maxVisited: 100000});
    if (weighted.path) return {...weighted, strategy: "Weighted Push A*"};
    return search({...payload, algorithm: "push-astar", maxVisited: 150000});
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
    ["greedy", "push-greedy"].includes(algorithm) ? heuristic(s.boxes, board.goals) :
    s.cost + weight * heuristic(s.boxes, board.goals);
  frontier.push([score(initial), order++, initial]);
  while (frontier.length) {
    const current = frontier.pop()[2];
    const signature = pushMacro ? pushKey(current, board) :
      key(current.robot, current.boxes.map(b => b.join(",")));
    if (seen.has(signature) && seen.get(signature) <= current.cost) continue;
    seen.set(signature, current.cost); visited++;
    if (goal(current.boxes, board.goals)) return {path: current.path, visited};
    if (payload.maxVisited && visited >= payload.maxVisited) return {path: null, visited, budgetHit: true};
    const nextStates = pushMacro ? pushNeighbors(current, board) :
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
onmessage = ({data}) => postMessage({type: "done", ...search(data)});
