const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {performance} = require("node:perf_hooks");
const {LEVELS, stateFromRows} = require("../docs/levels.js");
const {evaluateCheckpoints} = require("./evaluator.js");

const DIRS = {Up: [-1, 0], Down: [1, 0], Left: [0, -1], Right: [0, 1]};

function loadWorker(progress, streamProgress) {
  const source = fs.readFileSync(path.join(__dirname, "..", "docs", "solver-worker.js"), "utf8");
  const context = {
    postMessage: message => {
      progress.push(message);
      if (streamProgress) {
        process.stdout.write(`${JSON.stringify({type: "progress", message})}\n`);
      }
    },
    onmessage: null,
    console,
  };
  vm.runInNewContext(source, context, {filename: "solver-worker.js"});
  return context;
}

function parseRows(rows) {
  const board = {rows, walls: new Set(), goals: new Map(), floor: new Set()};
  const boxes = new Map();
  let robot = null;
  rows.forEach((row, y) => [...row].forEach((ch, x) => {
    const p = `${y},${x}`;
    if (ch === "O") board.walls.add(p);
    else board.floor.add(p);
    if (ch === "R") robot = [y, x];
    else if (ch === "X" || (/[A-Z]/.test(ch) && !"ORS".includes(ch))) boxes.set(p, ch);
    else if (ch === "S") board.goals.set(p, "X");
    else if (/[a-z]/.test(ch)) board.goals.set(p, ch.toUpperCase());
  }));
  return {board, robot, boxes};
}

function cloneState(state) {
  return {board: state.board, robot: [...state.robot], boxes: new Map(state.boxes)};
}

function isGoal(state) {
  return [...state.boxes].every(([position, label]) => state.board.goals.get(position) === label);
}

function moveState(state, direction) {
  const vector = DIRS[direction];
  if (!vector) return null;
  const [dy, dx] = vector;
  const [y, x] = state.robot;
  const next = `${y + dy},${x + dx}`;
  if (!state.board.floor.has(next)) return null;
  const result = cloneState(state);
  if (state.boxes.has(next)) {
    const beyond = `${y + 2 * dy},${x + 2 * dx}`;
    if (!state.board.floor.has(beyond) || state.boxes.has(beyond)) return null;
    const label = state.boxes.get(next);
    result.boxes.delete(next);
    result.boxes.set(beyond, label);
  }
  result.robot = [y + dy, x + dx];
  return result;
}

function validatePathToGoal(rows, pathResult) {
  if (!pathResult) return {valid: false, moves: 0, pushes: 0, reason: "missing-path"};
  const path = Array.from(pathResult);
  let state = parseRows(rows);
  let pushes = 0;
  for (let index = 0; index < path.length; index++) {
    const move = path[index];
    const before = new Set(state.boxes.keys());
    const next = moveState(state, move);
    if (!next) return {valid: false, moves: path.length, pushes, reason: `illegal-${move}`};
    state = next;
    for (const position of state.boxes.keys()) {
      if (!before.has(position)) {
        pushes++;
        break;
      }
    }
    if (isGoal(state)) return {valid: true, moves: index + 1, pushes};
  }
  return {valid: isGoal(state), moves: path.length, pushes,
    reason: isGoal(state) ? undefined : "not-solved"};
}

function progressSummary(messages) {
  const progress = messages.filter(message => message && message.type === "progress");
  const best = progress.reduce((winner, message) => {
    if (!Number.isFinite(message.bestEstimate)) return winner;
    if (!winner || message.bestEstimate < winner.bestEstimate ||
        (message.bestEstimate === winner.bestEstimate &&
          (message.bestPushes || 0) < (winner.bestPushes || 0))) {
      return message;
    }
    return winner;
  }, null);
  return {
    messages: messages.length,
    progressMessages: progress.length,
    last: progress.at(-1) || null,
    bestEstimate: best?.bestEstimate,
    bestPushes: best?.bestPushes,
  };
}

function runCase(caseSpec) {
  const levelRows = caseSpec.rows || LEVELS[caseSpec.level];
  if (!levelRows) throw new Error(`Unknown benchmark level: ${caseSpec.level}`);
  const progress = [];
  const worker = loadWorker(progress, Boolean(caseSpec.streamProgress));
  const payload = {
    algorithm: caseSpec.algorithm || "push-beam",
    state: stateFromRows(levelRows),
    ...(caseSpec.payload || {}),
  };
  const started = performance.now();
  const result = worker.search(payload);
  const elapsedMs = Math.round(performance.now() - started);
  const validation = validatePathToGoal(levelRows, result.path);
  const checkpointEvaluation = evaluateCheckpoints(levelRows, result);
  return {
    name: caseSpec.name || `${caseSpec.level}:${payload.algorithm}`,
    level: caseSpec.level || "custom",
    algorithm: payload.algorithm,
    elapsedMs,
    solved: validation.valid,
    valid: validation.valid || !result.path,
    moves: validation.valid ? validation.moves : 0,
    pushes: validation.valid ? validation.pushes : 0,
    visited: result.visited || 0,
    cutoff: Boolean(result.cutoff),
    terminationReason: result.terminationReason,
    bestEstimate: result.bestEstimate,
    bestPushes: result.bestPushes,
    checkpointCount: result.checkpoints?.length || 0,
    checkpointEvaluation,
    progress: progressSummary(progress),
    validation,
  };
}

try {
  const caseSpec = JSON.parse(process.argv[2] || "{}");
  const result = runCase(caseSpec);
  process.stdout.write(`${JSON.stringify(
    caseSpec.streamProgress ? {type: "result", result} : result,
  )}\n`);
} catch (error) {
  process.stdout.write(`${JSON.stringify(process.argv[2]?.includes("streamProgress")
    ? {type: "error", error: error.message, stack: error.stack}
    : {
    error: error.message,
    stack: error.stack,
  })}\n`);
  process.exitCode = 1;
}
