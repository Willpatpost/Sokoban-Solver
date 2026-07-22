const LEVELS = SokomindLevels.LEVELS;
const DIRS = {Up: [-1, 0], Down: [1, 0], Left: [0, -1], Right: [0, 1]};
const CODE_MOVE = {U: "Up", D: "Down", L: "Left", R: "Right"};
const SOLVER_BUILD = "2026-07-22.12";
const SOLVER_WORKER_URL = `solver-worker.js?build=${SOLVER_BUILD}`;
const PUSH_BOUNDS_KEY = "sokomind-push-bounds-v1";
const KEYS = {ArrowUp: "Up", ArrowDown: "Down", ArrowLeft: "Left", ArrowRight: "Right",
  w: "Up", W: "Up", s: "Down", S: "Down", a: "Left", A: "Left", d: "Right", D: "Right"};
const $ = (id) => document.getElementById(id);

let levelKey = "ultra-tiny", state, initialState, history = [], moveHistory = [], moves = 0;
let workers = [], animation = [], timer = null, solvedShown = false;
let startedAt = null, elapsed = 0, clock = null;
let pushBounds = {};
let searchLog = [], searchStartedAt = null;
let searchRunId = null, searchLogSequence = 0;
let searchTelemetryTimers = [];

try {
  const storedBounds = JSON.parse(localStorage.getItem(PUSH_BOUNDS_KEY) || "{}");
  for (const [key, value] of Object.entries(storedBounds)) {
    if (!Number.isInteger(value) || value <= 0) continue;
    pushBounds[key] = Math.min(pushBounds[key] ?? Infinity, value);
  }
} catch (_error) {
  // Storage can be unavailable in private browsing; search remains unbounded.
}

function rememberPushBound() {
  const pushes = moveHistory.filter(entry => entry.pushed).length;
  if (!pushes || pushes >= (pushBounds[levelKey] ?? Infinity)) return;
  pushBounds[levelKey] = pushes;
  try { localStorage.setItem(PUSH_BOUNDS_KEY, JSON.stringify(pushBounds)); } catch (_error) {}
}

function currentUpperBound() {
  return history.length === 0 ? pushBounds[levelKey] : undefined;
}

function planUpperBound(plan) {
  const incumbent = currentUpperBound();
  return Number.isFinite(incumbent) ? incumbent + (plan.boundSlack || 0) : incumbent;
}

function parse(rows) {
  const board = {rows, walls: new Set(), goals: new Map(), floor: new Set()};
  const boxes = new Map(); let robot;
  rows.forEach((row, y) => [...row].forEach((ch, x) => {
    const p = `${y},${x}`;
    if (ch === "O") board.walls.add(p); else board.floor.add(p);
    if (ch === "R") robot = [y, x];
    else if (ch === "X" || (/[A-Z]/.test(ch) && !"ORS".includes(ch))) boxes.set(p, ch);
    else if (ch === "S") board.goals.set(p, "X");
    else if (/[a-z]/.test(ch)) board.goals.set(p, ch.toUpperCase());
  }));
  return {board, robot, boxes};
}
const cloneState = (s) => ({board: s.board, robot: [...s.robot], boxes: new Map(s.boxes)});
const pos = (y, x) => `${y},${x}`;
function isGoal(s) {
  return [...s.boxes].every(([p, label]) => s.board.goals.get(p) === label);
}
function moveState(s, direction) {
  const [dy, dx] = DIRS[direction], [y, x] = s.robot;
  const next = pos(y + dy, x + dx);
  if (!s.board.floor.has(next)) return null;
  const result = cloneState(s);
  if (s.boxes.has(next)) {
    const beyond = pos(y + 2 * dy, x + 2 * dx);
    if (!s.board.floor.has(beyond) || s.boxes.has(beyond)) return null;
    const label = s.boxes.get(next);
    result.boxes.delete(next); result.boxes.set(beyond, label);
  }
  result.robot = [y + dy, x + dx];
  return result;
}
function isPushMove(s, direction) {
  const [dy, dx] = DIRS[direction], [y, x] = s.robot;
  return s.boxes.has(pos(y + dy, x + dx));
}
function moveHistoryText() {
  return moveHistory.map(({direction, pushed}, index) => (
    `${index + 1}. ${direction}${pushed ? " (push)" : ""}`
  )).join("\n");
}
function renderMoveHistory() {
  $("move-history-count").textContent = moves.toLocaleString();
  $("move-history-text").value = moveHistoryText();
}
function searchLogText(entries = searchLog) { return entries.map(entry => entry.text).join("\n"); }
function searchLogJsonLines(entries = searchLog) {
  return entries.map(entry => JSON.stringify(entry.event)).join("\n");
}
function structuredSearchStats(stats) {
  return Object.fromEntries(Object.entries(stats || {}).map(([name, value]) => {
    if (typeof value !== "string") return [name, value];
    const compact = value.replaceAll(",", "");
    if (/^-?\d+(?:\.\d+)?$/.test(compact)) return [name, Number(compact)];
    if (/^-?\d+(?:\.\d+)?\/s$/.test(compact)) return [name, Number(compact.slice(0, -2))];
    if (/^-?\d+(?:\.\d+)?s$/.test(compact)) return [name, Number(compact.slice(0, -1))];
    return [name, value];
  }));
}
function renderSearchLog() {
  $("search-log-count").textContent = searchLog.length.toLocaleString();
  const output = $("search-log-text");
  output.value = searchLogText(searchLog.slice(-1500));
  output.scrollTop = output.scrollHeight;
}
function resetSearchLog() {
  searchLog = [];
  searchStartedAt = performance.now();
  searchLogSequence = 0;
  searchRunId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  renderSearchLog();
}
function appendSearchLog(category, message, stats = null) {
  if (searchStartedAt === null) searchStartedAt = performance.now();
  const elapsedSeconds = (performance.now() - searchStartedAt) / 1000;
  const detail = stats ? Object.entries(stats)
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([name, value]) => `${name}=${value}`)
    .join(" ") : "";
  const text = `[${formatTime(elapsedSeconds)}] ${category.toUpperCase()}  ${message}${detail ? ` | ${detail}` : ""}`;
  const event = {
    schemaVersion: 1,
    runId: searchRunId,
    sequence: ++searchLogSequence,
    timestamp: new Date().toISOString(),
    elapsedMs: Math.round(elapsedSeconds * 1000),
    build: SOLVER_BUILD,
    level: levelKey,
    category,
    message,
    stats: structuredSearchStats(stats),
  };
  searchLog.push({text, category, elapsedSeconds, event});
  renderSearchLog();
}

function title(key) { return key.split("-").map(x => x[0].toUpperCase() + x.slice(1)).join(" "); }
function shortStateId(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}
function renderLevels() {
  $("level-count").textContent = `${Object.keys(LEVELS).length} puzzles`;
  $("level-list").replaceChildren(...Object.entries(LEVELS).map(([key, rows]) => {
    const button = document.createElement("button");
    button.className = `level-card${key === levelKey ? " active" : ""}`;
    const thumb = document.createElement("span"); thumb.className = "thumbnail";
    const grid = document.createElement("span"); grid.className = "thumbnail-grid";
    grid.style.gridTemplateColumns = `repeat(${rows[0].length}, 5px)`;
    rows.forEach(row => [...row].forEach(ch => {
      const cell = document.createElement("i");
      cell.className = "mini " + (ch === "O" ? "wall" : ch === "R" ? "robot" :
        (ch === "X" || (/[A-Z]/.test(ch) && !"ORS".includes(ch))) ? "box" :
        (ch === "S" || /[a-z]/.test(ch)) ? "goal" : "");
      grid.append(cell);
    }));
    thumb.append(grid);
    const copy = document.createElement("span");
    const name = document.createElement("strong");
    const size = document.createElement("small");
    name.textContent = title(key);
    size.textContent = `${rows[0].length} x ${rows.length}`;
    copy.append(name, size);
    button.append(thumb, copy); button.onclick = () => loadLevel(key);
    return button;
  }));
}
function render() {
  $("level-title").textContent = title(levelKey);
  $("move-count").textContent = moves;
  const board = $("board"), rows = state.board.rows;
  board.style.gridTemplateColumns = `repeat(${rows[0].length}, 1fr)`;
  board.style.setProperty("--cols", rows[0].length);
  board.style.setProperty("--rows", rows.length);
  fitBoardToScreen();
  const cells = [];
  rows.forEach((row, y) => [...row].forEach((_ch, x) => {
    const p = pos(y, x), cell = document.createElement("div");
    cell.className = `tile${state.board.walls.has(p) ? " wall" : ""}`;
    if (state.board.goals.has(p)) {
      cell.classList.add("goal");
      cell.dataset.goal = state.board.goals.get(p) === "X" ? "S" : state.board.goals.get(p).toLowerCase();
    }
    if (state.boxes.has(p)) {
      const piece = document.createElement("span"), label = state.boxes.get(p);
      piece.className = `piece box${state.board.goals.get(p) === label ? " done" : ""}`;
      piece.textContent = label; cell.append(piece);
    }
    if (state.robot[0] === y && state.robot[1] === x) {
      const piece = document.createElement("span"); piece.className = "piece robot";
      piece.textContent = "R"; cell.append(piece);
    }
    cells.push(cell);
  }));
  board.replaceChildren(...cells);
}
function fitBoardToScreen() {
  if (!state) return;
  const board = $("board"), wrap = $("board-wrap"), rows = state.board.rows;
  if (!board || !wrap || !window.matchMedia("(max-width: 700px)").matches) {
    board?.style.removeProperty("--tile-size");
    return;
  }
  const cols = rows[0].length, rowCount = rows.length;
  const wrapWidth = Math.max(240, wrap.clientWidth || window.innerWidth);
  const maxBoardHeight = Math.max(260, window.innerHeight * 0.48);
  const boardPadding = 16;
  const gap = 1;
  const byWidth = (wrapWidth - boardPadding * 2 - gap * (cols - 1)) / cols;
  const byHeight = (maxBoardHeight - boardPadding * 2 - gap * (rowCount - 1)) / rowCount;
  const size = Math.floor(Math.max(15, Math.min(34, byWidth, byHeight)));
  board.style.setProperty("--tile-size", `${size}px`);
}
function loadLevel(key) {
  stop(); levelKey = key; state = parse(LEVELS[key]); initialState = cloneState(state);
  history = []; moveHistory = []; moves = 0; solvedShown = false; resetTimer();
  setStatus("Use arrow keys or WASD to play."); renderLevels(); render();
  renderMoveHistory(); resetSearchLog();
}
function setControlsBusy(active) {
  $("solve").disabled = active;
  $("hint").disabled = active;
  $("algorithm").disabled = active;
}
function tryMove(direction, fromSolver = false) {
  const pushed = isPushMove(state, direction);
  const next = moveState(state, direction);
  if (!next) { if (!fromSolver) setStatus(`${direction} is blocked.`); return false; }
  startTimer();
  history.push(cloneState(state)); state = next; moves++;
  moveHistory.push({direction, pushed}); render(); renderMoveHistory();
  if (isGoal(state)) complete(); else if (!fromSolver) setStatus("Playing");
  return true;
}
function complete() {
  stop(false); setControlsBusy(false); freezeTimer(); setStatus(`Solved in ${moves} moves!`);
  rememberPushBound();
  if (solvedShown) return; solvedShown = true;
  $("complete-level").textContent = title(levelKey);
  $("complete-moves").textContent = moves;
  const keys = Object.keys(LEVELS), hasNext = keys.indexOf(levelKey) < keys.length - 1;
  $("next-level").hidden = !hasNext; $("complete-dialog").showModal();
}
function setStatus(text) { $("status").textContent = text; }
function undo() {
  stop(); if (!history.length) return;
  state = history.pop(); moveHistory.pop(); moves--; solvedShown = false;
  render(); renderMoveHistory(); setStatus("Undid one move.");
}
function reset() {
  stop(); state = cloneState(initialState); history = []; moveHistory = []; moves = 0; solvedShown = false;
  resetTimer();
  render(); renderMoveHistory(); setStatus("Level reset.");
}
function clearSearchTelemetry() {
  searchTelemetryTimers.forEach(handle => clearInterval(handle));
  searchTelemetryTimers = [];
}
function stop(message = true) {
  if (workers.length && message) appendSearchLog("control", "Search stopped by user",
    {activeWorkers: workers.length});
  workers.forEach(worker => worker.terminate()); workers = [];
  clearSearchTelemetry();
  animation = []; clearTimeout(timer); timer = null;
  setControlsBusy(false);
  if (message && state) setStatus("Stopped.");
}
function formatTime(seconds) {
  const whole = Math.floor(seconds), minutes = Math.floor(whole / 60);
  return `${String(minutes).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")}`;
}
function startTimer() {
  if (startedAt !== null) return;
  startedAt = Date.now() - elapsed * 1000;
  clock = setInterval(updateTimer, 250); updateTimer();
}
function updateTimer() {
  if (startedAt !== null) elapsed = (Date.now() - startedAt) / 1000;
  $("timer").textContent = formatTime(elapsed);
}
function freezeTimer() {
  updateTimer(); startedAt = null; clearInterval(clock); clock = null;
}
function resetTimer() {
  startedAt = null; elapsed = 0; clearInterval(clock); clock = null;
  $("timer").textContent = "00:00";
}
function showHome() {
  stop(false); freezeTimer(); $("home-screen").classList.remove("hidden");
}
function hideHome() {
  $("home-screen").classList.add("hidden"); $("board").focus();
}
function serializeState(s) {
  return {rows: s.board.rows, robot: s.robot, boxes: [...s.boxes]};
}
function validatePathToGoal(path) {
  return SokomindPath.validatePathToGoal(state, path, cloneState, moveState, isGoal);
}
function walkBetween(board, boxes, start, target) {
  const blocked = new Set(boxes.map(([y, x]) => pos(y, x)));
  const startKey = pos(start[0], start[1]), targetKey = pos(target[0], target[1]);
  const paths = new Map([[startKey, []]]), queue = [start];
  for (let head = 0; head < queue.length; head++) {
    const [y, x] = queue[head], path = paths.get(pos(y, x));
    if (pos(y, x) === targetKey) return path;
    for (const [move, [dy, dx]] of Object.entries(DIRS)) {
      const next = pos(y + dy, x + dx);
      if (paths.has(next) || !board.floor.has(next) || blocked.has(next)) continue;
      paths.set(next, [...path, move]);
      queue.push([y + dy, x + dx]);
    }
  }
  return null;
}
function boxesFromMeetKey(key) {
  const boxPart = key.split("|")[1] || "";
  if (!boxPart) return [];
  return boxPart.split(";").filter(Boolean).map(item => {
    const [y, x, label] = item.split(",");
    return [Number(y), Number(x), label];
  });
}
function decodeSegment(segment) {
  return typeof segment === "string" ? [...segment].map(code => CODE_MOVE[code]) : segment;
}
function reconstructMeetPath(meetKey, forwardSeen, reverseSeen) {
  const forwardSegments = [];
  let current = forwardSeen.get(meetKey);
  if (!current || !reverseSeen.has(meetKey)) return null;
  while (current?.parent) {
    forwardSegments.unshift(...decodeSegment(current.segment));
    current = forwardSeen.get(current.parent);
    if (!current) return null;
  }

  const reverseSegments = [];
  current = reverseSeen.get(meetKey);
  while (current?.parent) {
    reverseSegments.push(...decodeSegment(current.segment));
    current = reverseSeen.get(current.parent);
    if (!current) return null;
  }

  const forward = forwardSeen.get(meetKey), reverse = reverseSeen.get(meetKey);
  const bridge = walkBetween(state.board, boxesFromMeetKey(forward.id), forward.robot, reverse.robot);
  if (!bridge) return null;
  return [...forwardSegments, ...bridge, ...reverseSegments];
}
function checkpointMeetKey(checkpointState) {
  const boxes = new Set(checkpointState.boxes.map(([position]) => position));
  const start = pos(checkpointState.robot[0], checkpointState.robot[1]);
  const reached = new Set([start]), queue = [checkpointState.robot];
  for (let head = 0; head < queue.length; head++) {
    const [y, x] = queue[head];
    for (const [dy, dx] of Object.values(DIRS)) {
      const next = pos(y + dy, x + dx);
      if (reached.has(next) || boxes.has(next) || !state.board.floor.has(next)) continue;
      reached.add(next);
      queue.push([y + dy, x + dx]);
    }
  }
  const representative = [...reached].sort()[0];
  const boxPart = checkpointState.boxes
    .map(([position, label]) => `${position},${label}`)
    .sort()
    .join(";");
  return `${representative}|${boxPart}`;
}
function reconstructCheckpointMeetPath(checkpoint, prefixPath, reverseSeen) {
  const meetKey = checkpointMeetKey(checkpoint.state);
  let current = reverseSeen.get(meetKey);
  if (!current) return null;
  const boxes = checkpoint.state.boxes.map(([position]) => position.split(",").map(Number));
  const bridge = walkBetween(
    state.board,
    boxes,
    checkpoint.state.robot,
    current.robot,
  );
  if (!bridge) return null;
  const suffix = [];
  while (current?.parent) {
    suffix.push(...decodeSegment(current.segment));
    current = reverseSeen.get(current.parent);
    if (!current) return null;
  }
  return [...prefixPath, ...bridge, ...suffix];
}
function solverPlans(algorithm) {
  if (!["ultimate", "portfolio", "fast"].includes(algorithm)) {
    return [{algorithm, label: $("algorithm").selectedOptions[0].text}];
  }
  return [
    {algorithm: "push-beam", label: "Push Beam", beamWidth: 1800, weight: 3, seed: 17},
    {algorithm: "push-greedy", label: "Push Greedy"},
    {algorithm: "weighted-push-astar", label: "Weighted Push A*"},
    {algorithm: "push-astar", label: "Push A*"},
  ];
}
function startBidirectionalSolver(purpose) {
  stop(false); setControlsBusy(true);
  resetSearchLog();
  setStatus(`Ultimate Bidirectional ${SOLVER_BUILD} is analyzing the puzzle...`);
  appendSearchLog("director", "Reading puzzle before worker allocation",
    {level: levelKey, boxes: state.boxes.size, floor: state.board.floor.size});
  const planner = new Worker(SOLVER_WORKER_URL);
  workers.push(planner);
  planner.onmessage = ({data}) => {
    if (data.type !== "done") return;
    planner.terminate();
    workers = workers.filter(worker => worker !== planner);
    if (!data.analysis) {
      setControlsBusy(false); setStatus("Puzzle analysis failed.");
      appendSearchLog("error", "Planner returned no analysis");
      return;
    }
    runBidirectionalSolver(purpose, data.analysis);
  };
  planner.onerror = () => {
    planner.terminate(); workers = workers.filter(worker => worker !== planner);
    setControlsBusy(false); setStatus("Puzzle analysis worker failed.");
    appendSearchLog("error", "Puzzle analysis worker failed");
  };
  planner.postMessage({algorithm: "analyze-puzzle", state: serializeState(state)});
}
function runBidirectionalSolver(purpose, analysis) {
  setStatus(`Ultimate Bidirectional ${SOLVER_BUILD} is searching...`);
  const hardware = navigator.hardwareConcurrency || 2;
  const recommendations = analysis.recommendations;
  const searchScale = analysis.searchScale;
  const advancedPortfolio = ["complex", "extreme"].includes(analysis.difficulty);
  const reverseLimit = recommendations.reverseWorkerLimit;
  const sideVisitedLimit = recommendations.sideVisitedLimit;
  const reverseWorkers = Math.max(1, Math.min(
    hardware - 1,
    reverseLimit,
    Math.max(1, analysis.reverseStartPulls || analysis.reverseStartRegions || 1),
  ));
  appendSearchLog("analysis", `${analysis.difficulty} puzzle analyzed`, {
    size: `${analysis.dimensions.columns}x${analysis.dimensions.rows}`,
    boxes: analysis.boxes,
    h0: analysis.initialHeuristic,
    pushes: analysis.legalPushes,
    rooms: analysis.rooms.length,
    gates: analysis.articulations,
    tunnels: analysis.tunnelCells,
    surplus: analysis.surplusBoxes,
    pressure: analysis.pressure,
    reverseRegions: analysis.reverseStartRegions,
    productiveReverseRegions: analysis.productiveReverseStartRegions,
    reversePulls: analysis.reverseStartPulls,
  });
  analysis.phases.forEach((phase, index) => appendSearchLog(
    "plan", `Phase ${index + 1}: ${phase.id}`, {reason: phase.reason},
  ));
  analysis.rooms.forEach((room, index) => appendSearchLog(
    "topology", `Gated room ${index + 1}`, {
      gate: room.gate, cells: room.cells, boxes: room.boxes, goals: room.goals,
      surplus: room.surplus, dependencies: room.dependencies, depth: room.maxDepth,
    },
  ));
  appendSearchLog("director", "Allocated initial portfolio", {
    hardwareThreads: hardware,
    reverseWorkers,
    beamAttempts: recommendations.beamAttempts,
    beamWidth: recommendations.beamWidth,
    preparedBoard: Boolean(analysis.preparedBoard),
  });
  const beamProfiles = [
    {beamProfile: "balanced", weight: 3, diversity: 1.75,
      topologyWeight: 0.8, goalPackingWeight: 0.8},
    {beamProfile: "detour", weight: 2, diversity: 2.5,
      topologyWeight: 1.4, goalPackingWeight: 1.1},
    {beamProfile: "milestone", weight: 2.3, diversity: 2,
      topologyWeight: 2, goalPackingWeight: 1.3},
  ];
  const beamAttemptCount = recommendations.beamAttempts;
  const beamPlans = Array.from({length: beamAttemptCount}, (_item, index) => ({
    algorithm: "push-beam",
    side: "direct",
    label: `Beam Restart ${index + 1}/${beamAttemptCount}`,
    beamWidth: recommendations.beamWidth,
    maxDepth: 600,
    maxVisited: recommendations.beamVisited,
    transpositionLimit: analysis.difficulty === "extreme" ? 18000 : 45000,
    seed: 29 + index * 104729,
    phaseScope: "opening",
    ...beamProfiles[index % beamProfiles.length],
  }));
  const primaryMacroProfile = recommendations.useEvacuation
    ? {beamProfile: "evacuation", weight: 1, topologyWeight: 1,
      evacuationWeight: 8, goalPackingWeight: 1, diversity: 1.5,
      seed: 29, beamWidth: 50, sequenceMacroResults: 8,
      continuationVisited: 0, endgameVisited: 0, handoffStage: "evacuation"}
    : {beamProfile: "milestone", weight: 2.4, topologyWeight: 1.6,
      goalPackingWeight: 1.3, diversity: 2, seed: 29, beamWidth: 50,
      sequenceMacroResults: 8, handoffStage: "packing"};
  const macroProfiles = [
    primaryMacroProfile,
    {beamProfile: "detour", weight: 3.2, topologyWeight: 0.7,
      goalPackingWeight: 1.5, diversity: 1.2, seed: 104800, beamWidth: 50,
      sequenceMacroResults: 6},
    {beamProfile: "detour", weight: 3, topologyWeight: 0.9,
      goalPackingWeight: 1.7, diversity: 1.8, seed: 314258, beamWidth: 50,
      sequenceMacroResults: 8},
    {beamProfile: "milestone", weight: 2.5, topologyWeight: 1.1,
      goalPackingWeight: 1.4, diversity: 2.2, seed: 418987, beamWidth: 50,
      sequenceMacroResults: 8},
  ];
  const macroPlans = advancedPortfolio && recommendations.useSequenceMacros ? macroProfiles.map((settings, index) => ({
    algorithm: "push-beam",
    side: "direct",
    label: `Box-Run Macro ${index + 1}/${macroProfiles.length}`,
    maxDepth: 600,
    maxVisited: 4000,
    transpositionLimit: 16000,
    sequenceMacros: true,
    sequenceMacroLimit: 12,
    sequenceMacroExplored: 32,
    endgameVisited: 60000,
    endgameThreshold: 60,
    endgameCandidates: 32,
    endgameAttempts: 16,
    endgameProfiles: ["balanced", "detour", "room-flow"],
    boundSlack: 80,
    continuationVisited: 20000,
    continuationAttempts: 16,
    continuationWidth: 36,
    continuationProfiles: [
      {beamProfile: "detour", weight: 3.5, topologyWeight: 0.6,
        goalPackingWeight: 1.7, diversity: 1.4},
      {beamProfile: "milestone", weight: 2.6, topologyWeight: 1.2,
        goalPackingWeight: 1.4, diversity: 2},
      {beamProfile: "balanced", weight: 4, topologyWeight: 0.35,
        goalPackingWeight: 1.8, diversity: 1.1},
    ],
    phaseScope: index === 0 && recommendations.useEvacuation ? "evacuation" : "opening",
    ...settings,
  })) : [];
  const dfsProfiles = [
    {dfsProfile: "setup", discrepancyLimit: 0, maxVisited: 20000},
    {dfsProfile: "setup", discrepancyLimit: 2, maxVisited: 80000},
  ];
  const dfsPlans = advancedPortfolio ? dfsProfiles.map((settings, index) => ({
    algorithm: "bounded-push-dfs",
    side: "direct",
    label: `Discrepancy DFS ${index + 1}/${dfsProfiles.length}`,
    maxDepth: 600,
    transpositionLimit: 60000,
    diversity: 1.5,
    seed: 313337 + index * 104729,
    phaseScope: "opening",
    ...settings,
  })) : [];
  macroPlans.forEach(plan => { plan.checkpointLimit = recommendations.checkpointLimit; });
  const evacuationPlans = macroPlans.filter(plan => plan.phaseScope === "evacuation");
  const remainingMacroPlans = macroPlans.filter(plan => plan.phaseScope !== "evacuation");
  let directOrder = 0;
  const directQueue = [...evacuationPlans, ...beamPlans, ...remainingMacroPlans, ...dfsPlans]
    .map(plan => ({...plan, queuePriority: plan.phaseScope === "evacuation" ? 0 : 50,
      queueOrder: directOrder++}));
  const maxWorkerConcurrency = Math.max(2, Math.min(4, hardware));
  const bridgeMaxOutstanding = Math.min(4, Math.max(1, maxWorkerConcurrency - 1));
  let activeDirectWorkers = 0, activeEvacuationWorkers = 0, activeBridgeWorkers = 0;
  let activeSideWorkers = 0;
  let lastQueuedDirectKind = null;
  const forwardRecords = new Map(), reverseRecordSets = [];
  const directCheckpoints = new Map();
  const reverseLandmarks = [], bridgePairs = new Set();
  const bridgeCampaignViable = new Map(), bridgeCampaignOutstanding = new Map();
  const bridgeCampaignTracker = SokomindDirectorPolicy.createBridgeCampaignTracker();
  const loggedExhaustedCampaigns = new Set();
  let bridgeOutstanding = 0, bridgeSerial = 0;
  const workerSides = new Map(), workerRecords = new Map(), workerProgress = new Map();
  const workerStarted = new Map(), workerLastMessage = new Map(), workerTelemetry = new Map();
  const workerPlans = new Map(), workerCancellations = new Map();
  let expectedWorkers = reverseWorkers + 1 + directQueue.length;
  const requiredWork = SokomindDirectorPolicy.createRequiredWorkTracker(expectedWorkers);
  let settled = false, totalVisited = 0, doneWorkers = 0;
  let targetedReverseQueued = false, exactShardsStarted = false, exactShardsExhausted = 0;
  let discardedExactIncumbent = false;
  const completedPhases = [];

  const isRequiredPlan = plan => plan.handoffStage !== "bridge" &&
    !plan.persistentExact && !plan.anytimeGuided;

  const enqueueDirectPlans = (plans, {priority = 10} = {}) => {
    if (!plans.length) return;
    const numericPriority = priority === false ? 100 : priority === true ? 10 : priority;
    directQueue.push(...plans.map(plan => ({...plan,
      queuePriority: plan.queuePriority ?? numericPriority, queueOrder: directOrder++})));
    directQueue.sort((left, right) =>
      left.queuePriority - right.queuePriority || left.queueOrder - right.queueOrder);
    expectedWorkers += plans.length;
    requiredWork.schedule(plans.filter(isRequiredPlan).length);
    appendSearchLog("director", "Queued follow-up workers", {
      count: plans.length, priority: numericPriority,
      labels: plans.map(plan => plan.label).join(", "),
    });
    pumpDirectPlans();
  };

  const retirePendingPlans = (predicate, reason) => {
    const retired = directQueue.filter(predicate);
    if (!retired.length) return;
    directQueue.splice(0, directQueue.length, ...directQueue.filter(plan => !predicate(plan)));
    expectedWorkers -= retired.length;
    requiredWork.retire(retired.filter(isRequiredPlan).length);
    for (const plan of retired.filter(plan => plan.handoffStage === "bridge")) {
      bridgeOutstanding = Math.max(0, bridgeOutstanding - 1);
      bridgeCampaignOutstanding.set(plan.bridgeCampaign,
        Math.max(0, (bridgeCampaignOutstanding.get(plan.bridgeCampaign) || 0) - 1));
    }
    appendSearchLog("director", "Retired stale pending workers", {
      count: retired.length, reason,
      labels: retired.map(plan => plan.label).join(", "),
    });
  };

  const retireActiveWorkers = (predicate, reason) => {
    const targets = [...workerPlans].filter(([, plan]) => predicate(plan));
    for (const [worker] of targets) workerCancellations.get(worker)?.(reason);
    if (targets.length) appendSearchLog("director", "Retired active workers", {
      count: targets.length,
      reason,
      labels: targets.map(([, plan]) => plan.label).join(", "),
    });
  };

  const queueBridgePlans = () => {
    if (exactShardsStarted || !reverseLandmarks.length ||
        bridgeOutstanding >= bridgeMaxOutstanding) return;
    const candidates = [];
    for (const [checkpointKey, checkpointRecord] of directCheckpoints) {
      if (!checkpointRecord.bridgeEligible) continue;
      const totalBound = checkpointRecord.totalBound ?? Infinity;
      for (const entry of reverseLandmarks) {
        const campaign = `${entry.generation}|${checkpointRecord.generation}|${checkpointKey}`;
        if (!bridgeCampaignTracker.canSchedule(campaign)) continue;
        const pairKey = `${checkpointKey}=>${entry.generation}:${entry.landmark.id}`;
        if (bridgePairs.has(pairKey)) continue;
        const remaining = totalBound - checkpointRecord.pushCost - entry.landmark.cost;
        if (remaining <= 0) continue;
        candidates.push({entry, checkpointKey, checkpointRecord, totalBound, remaining,
          campaign, pairKey, projected: checkpointRecord.pushCost +
            entry.landmark.cost + entry.landmark.estimate});
      }
    }
    candidates.sort((left, right) => left.entry.priority - right.entry.priority ||
      left.projected - right.projected || left.pairKey.localeCompare(right.pairKey));
    const plans = [];
    const plannedByCampaign = new Map();
    for (const candidate of candidates) {
      if (bridgeOutstanding + plans.length >= bridgeMaxOutstanding) break;
      const viable = bridgeCampaignViable.get(candidate.campaign) || 0;
      const outstanding = bridgeCampaignOutstanding.get(candidate.campaign) || 0;
      const planned = plannedByCampaign.get(candidate.campaign) || 0;
      if (viable + outstanding + planned >= bridgeCampaignTracker.limits.maxProductive ||
          !bridgeCampaignTracker.canSchedule(candidate.campaign)) continue;
      const {entry, checkpointRecord} = candidate;
      bridgePairs.add(candidate.pairKey);
      bridgeCampaignTracker.recordScheduled(candidate.campaign);
      plannedByCampaign.set(candidate.campaign, planned + 1);
      plans.push({
        algorithm: "bridge-astar",
        side: "direct",
        label: `Landmark Bridge ${++bridgeSerial}`,
        handoffStage: "bridge",
        bridgePairKey: candidate.pairKey,
        bridgeCampaign: candidate.campaign,
        landmarkGeneration: entry.generation,
        state: checkpointRecord.checkpoint.state,
        prefixPath: checkpointRecord.prefixPath,
        prefixCost: checkpointRecord.pushCost,
        totalBound: candidate.totalBound,
        targetState: entry.landmark.state,
        targetId: entry.landmark.id,
        reverseRecordIndex: entry.reverseRecordIndex,
        reverseCost: entry.landmark.cost,
        upperBound: candidate.remaining,
        maxVisited: 60000,
        frontierLimit: 4000,
        weight: 1.35,
        forcedMacros: false,
        sourceCheckpointId: candidate.checkpointKey,
      });
    }
    bridgeOutstanding += plans.length;
    for (const plan of plans) bridgeCampaignOutstanding.set(plan.bridgeCampaign,
      (bridgeCampaignOutstanding.get(plan.bridgeCampaign) || 0) + 1);
    if (plans.length) appendSearchLog("bridge", "Candidate landmark bridges queued", {
      count: plans.length, outstanding: bridgeOutstanding,
      campaigns: new Set(plans.map(plan => plan.bridgeCampaign)).size,
    });
    enqueueDirectPlans(plans);
  };

  const registerLandmarks = (landmarks, worker, sourcePlan) => {
    const records = workerRecords.get(worker);
    const reverseRecordIndex = reverseRecordSets.indexOf(records);
    if (reverseRecordIndex < 0) return;
    const generation = sourcePlan.landmarkGeneration || `reverse-${reverseRecordIndex}`;
    for (const landmark of landmarks) {
      if (!landmark?.id || !landmark.state ||
          reverseLandmarks.some(entry => entry.reverseRecordIndex === reverseRecordIndex &&
            entry.landmark.id === landmark.id)) continue;
      reverseLandmarks.push({
        landmark, reverseRecordIndex, generation,
        priority: sourcePlan.targetedReverse ? 0 : 1,
      });
    }
    appendSearchLog("landmarks", "Reverse worker published landmarks", {
      received: landmarks.length, retained: reverseLandmarks.length, generation,
    });
    if (sourcePlan.targetedReverse) {
      retirePendingPlans(
        plan => plan.handoffStage === "bridge" &&
          String(plan.landmarkGeneration).startsWith("initial-"),
        "milestone landmarks superseded initial bridge targets",
      );
    }
    queueBridgePlans();
  };

  const queueCheckpointWorkers = (plan, data) => {
    const totalBound = plan.totalBound ?? planUpperBound(plan) ?? Infinity;
    const prefixPath = plan.prefixPath || [];
    const prefixCost = plan.prefixCost || 0;
    if (plan.handoffStage === "evacuation") {
      const checkpoint = data.phaseCheckpoint;
      if (!checkpoint) return;
      const consumed = prefixCost + checkpoint.cost;
      const remaining = totalBound - consumed;
      if (remaining <= 0) return;
      if (!targetedReverseQueued) {
        targetedReverseQueued = true;
        expectedWorkers++;
        requiredWork.schedule();
        launch({
          mode: "bidir-reverse",
          side: "reverse",
          label: "Milestone Reverse Search",
          landmarkGeneration: "evacuation-milestone",
          targetedReverse: true,
          state: checkpoint.state,
          maxVisited: 100000,
          frontierLimit: 40000,
          landmarkLimit: 256,
          reverseShard: {index: 0, count: 1},
        });
        appendSearchLog("director", "Evacuation phase triggered milestone reverse search", {
          g: consumed,
        });
      }
      enqueueDirectPlans([{
        algorithm: "push-beam",
        side: "direct",
        label: "Topology Packing Handoff",
        state: checkpoint.state,
        prefixPath: [...prefixPath, ...checkpoint.path],
        prefixCost: consumed,
        totalBound,
        upperBound: remaining,
        maxDepth: remaining,
        maxVisited: 2000,
        transpositionLimit: 24000,
        beamProfile: "milestone",
        weight: 2,
        topologyWeight: 1.2,
        goalPackingWeight: 1.4,
        diversity: 2,
        seed: 104800,
        beamWidth: 80,
        sequenceMacros: true,
        sequenceMacroLimit: 16,
        sequenceMacroExplored: 64,
        sequenceMacroResults: 10,
        checkpointLimit: 8,
        continuationVisited: 0,
        endgameVisited: 0,
        handoffStage: "packing",
      }], {priority: 0});
      return;
    }
    if (plan.handoffStage === "packing") {
      const checkpointLimit = analysis.difficulty === "extreme" ? 2 : 4;
      const checkpoints = (data.checkpoints || [])
        .sort((left, right) =>
          (prefixCost + left.cost + left.estimate) -
          (prefixCost + right.cost + right.estimate))
        .slice(0, checkpointLimit);
      if (checkpoints.length) retirePendingPlans(
        pendingPlan => pendingPlan.phaseScope === "opening" ||
          pendingPlan.handoffStage === "bridge",
        "packing checkpoint superseded opening and bridge exploration",
      );
      if (checkpoints.length) retireActiveWorkers(
        activePlan => activePlan.handoffStage === "bridge",
        "packing checkpoint reserved capacity for solution discovery",
      );
      if (recommendations.useMilestoneReverse && !targetedReverseQueued && checkpoints.length) {
        const checkpoint = checkpoints[0];
        targetedReverseQueued = true;
        expectedWorkers++;
        requiredWork.schedule();
        launch({
          mode: "bidir-reverse",
          side: "reverse",
          label: "Packing Milestone Reverse Search",
          landmarkGeneration: "packing-milestone",
          targetedReverse: true,
          state: checkpoint.state,
          maxVisited: recommendations.sideVisitedLimit,
          frontierLimit: 40000,
          landmarkLimit: 256,
          reverseShard: {index: 0, count: 1},
        });
        appendSearchLog("director", "Packing checkpoint triggered milestone reverse search", {
          g: prefixCost + checkpoint.cost,
          h: checkpoint.estimate,
          f: prefixCost + checkpoint.cost + checkpoint.estimate,
        });
      }
      const exactPlans = checkpoints.map((checkpoint, index) => {
        const consumed = prefixCost + checkpoint.cost;
        const remaining = totalBound - consumed;
        const profiles = ["room-flow", "balanced", "setup"];
        const profile = profiles[index % profiles.length];
        const localSlack = profile === "setup" ? 30 : profile === "room-flow" ? 40 : 50;
        const heuristicCeiling = Number.isFinite(checkpoint.estimate)
          ? checkpoint.estimate + localSlack : 180;
        const remainingCeiling = Number.isFinite(remaining) ? remaining : 180;
        const localBound = Math.min(remainingCeiling, heuristicCeiling, 180);
        return {
          algorithm: "bounded-push-dfs",
          side: "direct",
          label: `Exact Handoff ${index + 1}/${checkpoints.length}`,
          state: checkpoint.state,
          prefixPath: [...prefixPath, ...checkpoint.path],
          prefixCost: consumed,
          totalBound,
          upperBound: localBound,
          maxDepth: localBound,
          maxVisited: 100000,
          transpositionLimit: 100000,
          checkpointLimit: 8,
          dfsProfile: profile,
          discrepancyLimit: profile === "setup" ? 4 : undefined,
          forcedMacros: false,
          seed: 313337 + index * 104729,
          handoffStage: "exact",
        };
      }).filter(next => next.upperBound > 0);
      enqueueDirectPlans(exactPlans, {priority: 10});
    }
  };

  const finish = (path, strategy = "Bidirectional") => {
    const validated = validatePathToGoal(path);
    if (validated === null) return false;
    settled = true;
    appendSearchLog("solution", `${strategy} produced a replay-validated solution`, {
      moves: validated.length, states: totalVisited.toLocaleString(),
    });
    workers.forEach(worker => worker.terminate()); workers = [];
    clearSearchTelemetry();
    setControlsBusy(false);
    if (purpose === "hint") {
      setStatus(validated.length
        ? `Hint: ${validated[0]} - ${validated.length} moves remain (${strategy})`
        : "This puzzle is already solved.");
    } else {
      setStatus(`${strategy} found ${validated.length} moves after ${totalVisited.toLocaleString()} states.`);
      animation = validated; animate();
    }
    return true;
  };

  const requestSolve = (meetKey, reverseRecords) => {
    if (settled) return false;
    const path = reconstructMeetPath(meetKey, forwardRecords, reverseRecords);
    if (!path) return false;
    appendSearchLog("meeting", "Forward and reverse canonical states matched");
    return finish(path);
  };

  const requestCheckpointSolve = (meetKey, reverseRecords) => {
    const checkpoint = directCheckpoints.get(meetKey);
    if (!checkpoint) return false;
    const path = reconstructCheckpointMeetPath(
      checkpoint.checkpoint,
      checkpoint.prefixPath,
      reverseRecords,
    );
    if (path) appendSearchLog("meeting", "Direct checkpoint matched reverse frontier", {
      label: checkpoint.label,
    });
    return path ? finish(path, checkpoint.label) : false;
  };

  const registerCheckpoints = (plan, data) => {
    const prefixPath = plan.prefixPath || [];
    const candidates = [data.phaseCheckpoint, ...(data.checkpoints || [])]
      .filter(Boolean);
    for (const checkpoint of candidates) {
      const meetKey = checkpointMeetKey(checkpoint.state);
      const record = {
        checkpoint,
        prefixPath: [...prefixPath, ...(checkpoint.path || [])],
        pushCost: (plan.prefixCost || 0) + checkpoint.cost,
        totalBound: plan.totalBound ?? planUpperBound(plan) ?? Infinity,
        label: `${plan.label} + Reverse Frontier`,
        generation: plan.handoffStage || plan.label,
        bridgeEligible: ["evacuation", "packing"].includes(plan.handoffStage),
      };
      const existing = directCheckpoints.get(meetKey);
      if (existing && existing.pushCost <= record.pushCost) continue;
      directCheckpoints.set(meetKey, record);
      appendSearchLog("checkpoint", `${plan.label} published a replayable checkpoint`, {
        id: shortStateId(meetKey),
        g: record.pushCost,
        h: checkpoint.estimate,
        f: Number.isFinite(checkpoint.estimate)
          ? record.pushCost + checkpoint.estimate : undefined,
        reverseFrontiers: reverseRecordSets.length,
      });
      for (const reverseRecords of reverseRecordSets) {
        if (reverseRecords.has(meetKey) && requestCheckpointSolve(meetKey, reverseRecords)) {
          return true;
        }
      }
      queueBridgePlans();
    }
    return false;
  };

  const inspectRecords = (records, worker) => {
    const side = workerSides.get(worker);
    const recordMap = workerRecords.get(worker);
    const meetings = [];
    for (const record of records) {
      recordMap.set(record.id, record);
      if (side === "forward") {
        for (const reverseRecords of reverseRecordSets) {
          if (reverseRecords.has(record.id)) meetings.push([record.id, reverseRecords]);
        }
      } else if (forwardRecords.has(record.id)) {
        meetings.push([record.id, recordMap]);
      }
      if (side === "reverse" && directCheckpoints.has(record.id) &&
          requestCheckpointSolve(record.id, recordMap)) return;
    }
    for (const [meetKey, reverseRecords] of meetings) {
      if (requestSolve(meetKey, reverseRecords)) return;
    }
  };

  let exactRound = 0, exactRoundShardCount = 0;
  const launchAnytimeDiscovery = (limit) => {
    if (limit <= 0) return 0;
    const candidates = [...directCheckpoints]
      .map(([id, record]) => ({id, ...record}))
      .filter(candidate => !Number.isFinite(candidate.totalBound) ||
        candidate.pushCost + candidate.checkpoint.estimate <= candidate.totalBound);
    const selected = SokomindDirectorPolicy.selectAnytimeCheckpoints(candidates, limit);
    const profiles = [
      {beamProfile: "milestone", weight: 2.1, topologyWeight: 1.8,
        goalPackingWeight: 1.5, diversity: 2.2, seed: 15485863},
      {beamProfile: "detour", weight: 1.8, topologyWeight: 0.7,
        goalPackingWeight: 1.8, diversity: 2.8, seed: 32452843},
    ];
    if (!selected.length) return 0;
    expectedWorkers += selected.length;
    appendSearchLog("director", "Started anytime checkpoint discovery", {
      workers: selected.length,
      candidates: selected.map(candidate =>
        `${shortStateId(candidate.id)}:g${candidate.pushCost}:h${candidate.checkpoint.estimate}:f${candidate.projectedCost}`
      ).join(","),
    });
    selected.forEach((candidate, index) => {
      const remaining = Number.isFinite(candidate.totalBound)
        ? candidate.totalBound - candidate.pushCost : 320;
      launch({
        algorithm: "push-beam",
        side: "direct",
        label: `Anytime Guided ${index + 1}/${selected.length}`,
        anytimeGuided: true,
        handoffStage: "anytime",
        state: candidate.checkpoint.state,
        prefixPath: candidate.prefixPath,
        prefixCost: candidate.pushCost,
        totalBound: candidate.totalBound,
        upperBound: remaining,
        maxDepth: Math.min(320, remaining),
        maxVisited: 600000,
        beamWidth: 700,
        transpositionLimit: 120000,
        costWeight: 1.25,
        sequenceMacros: true,
        sequenceMacroLimit: 12,
        sequenceMacroExplored: 48,
        sequenceMacroResults: 10,
        checkpointLimit: 16,
        continuationVisited: 150000,
        continuationAttempts: 12,
        continuationWidth: 80,
        endgameVisited: 150000,
        endgameThreshold: 55,
        endgameAttempts: 12,
        endgameProfiles: ["room-flow", "balanced", "setup"],
        progressInterval: 25000,
        ...profiles[index % profiles.length],
      });
    });
    return selected.length;
  };

  const launchExactShards = () => {
    if (exactShardsStarted) return;
    exactShardsStarted = true;
    exactShardsExhausted = 0;
    retirePendingPlans(plan => plan.handoffStage === "bridge",
      "persistent exact search reserved proof capacity");
    const availableSlots = Math.max(1,
      maxWorkerConcurrency - activeSideWorkers - activeDirectWorkers);
    const anytimeWorkers = launchAnytimeDiscovery(Math.max(0, Math.min(2, availableSlots - 1)));
    exactRoundShardCount = anytimeWorkers
      ? 1 : Math.max(1, Math.min(3, hardware, availableSlots));
    const round = ++exactRound;
    const exactBound = discardedExactIncumbent ? Infinity : currentUpperBound() ?? Infinity;
    expectedWorkers += exactRoundShardCount;
    appendSearchLog("director", "Started persistent partitioned exact contour", {
      round, shards: exactRoundShardCount, shardDepth: 4, anytimeWorkers,
      bound: Number.isFinite(exactBound) ? exactBound : "unbounded",
      transpositionsPerShard: "160,000",
    });
    for (let index = 0; index < exactRoundShardCount; index++) {
      launch({
        algorithm: "push-ida-star",
        side: "direct",
        label: `Persistent Exact Shard ${index + 1}/${exactRoundShardCount}`,
        exhaustiveFallback: true,
        persistentExact: true,
        exactRound: round,
        exactShard: {index, count: exactRoundShardCount, depth: 4},
        upperBound: exactBound,
        maxVisited: Number.MAX_SAFE_INTEGER,
        transpositionLimit: 160000,
        progressInterval: 25000,
        forcedMacros: false,
        seed: 911 + index * 104729,
      });
    }
  };

  function pumpDirectPlans() {
    if (settled || activeEvacuationWorkers > 0) return;
    const directCapacity = Math.max(1,
      Math.min(2, maxWorkerConcurrency - activeSideWorkers));
    while (directQueue.length && activeDirectWorkers < directCapacity) {
      const bestPriority = directQueue[0].queuePriority;
      const eligible = directQueue
        .map((plan, index) => ({plan, index}))
        .filter(({plan}) => plan.queuePriority === bestPriority);
      let selected = eligible[0];
      const nonBridge = eligible.find(({plan}) => plan.handoffStage !== "bridge");
      if (nonBridge && (activeBridgeWorkers > 0 || lastQueuedDirectKind === "bridge")) {
        selected = nonBridge;
      }
      if (selected.plan.handoffStage === "bridge" && activeBridgeWorkers >= 1) {
        const requiredAlternative = directQueue
          .map((plan, index) => ({plan, index}))
          .find(({plan}) => plan.handoffStage !== "bridge");
        if (!requiredAlternative) break;
        selected = requiredAlternative;
      }
      const next = directQueue.splice(selected.index, 1)[0];
      lastQueuedDirectKind = next.handoffStage === "bridge" ? "bridge" : "other";
      launch(next);
      if (next.phaseScope === "evacuation") break;
    }
  }

  const launch = (plan) => {
    const worker = new Worker(SOLVER_WORKER_URL);
    const effectiveUpperBound = plan.upperBound ?? planUpperBound(plan);
    let workerFinished = false;
    workers.push(worker);
    workerPlans.set(worker, plan);
    if (plan.side === "direct") activeDirectWorkers++;
    if (["forward", "reverse"].includes(plan.side)) activeSideWorkers++;
    if (plan.handoffStage === "bridge") activeBridgeWorkers++;
    if (plan.phaseScope === "evacuation") activeEvacuationWorkers++;
    workerStarted.set(worker, performance.now());
    workerLastMessage.set(worker, performance.now());
    workerSides.set(worker, plan.side);
    appendSearchLog("worker", `Started ${plan.label}`, {
      role: plan.side,
      algorithm: plan.mode || plan.algorithm,
      budget: plan.persistentExact ? "persistent" : plan.maxVisited?.toLocaleString(),
      width: plan.beamWidth || plan.frontierLimit,
      profile: plan.beamProfile || plan.dfsProfile,
      bound: Number.isFinite(effectiveUpperBound) ? effectiveUpperBound : "unbounded",
      shard: plan.exactShard
        ? `${plan.exactShard.index + 1}/${plan.exactShard.count}`
        : plan.reverseShard
          ? `${plan.reverseShard.index + 1}/${plan.reverseShard.count}` : undefined,
      round: plan.exactRound,
      checkpoint: plan.sourceCheckpointId ? shortStateId(plan.sourceCheckpointId) : undefined,
    });
    const releaseWorker = ({queueBridge = true} = {}) => {
      worker.terminate();
      workers = workers.filter(item => item !== worker);
      workerSides.delete(worker);
      workerProgress.delete(worker);
      workerRecords.delete(worker);
      workerStarted.delete(worker);
      workerLastMessage.delete(worker);
      clearInterval(workerTelemetry.get(worker));
      workerTelemetry.delete(worker);
      workerPlans.delete(worker);
      workerCancellations.delete(worker);
      if (plan.side === "direct") activeDirectWorkers = Math.max(0, activeDirectWorkers - 1);
      if (["forward", "reverse"].includes(plan.side)) {
        activeSideWorkers = Math.max(0, activeSideWorkers - 1);
      }
      if (plan.handoffStage === "bridge") {
        activeBridgeWorkers = Math.max(0, activeBridgeWorkers - 1);
      }
      if (plan.phaseScope === "evacuation") {
        activeEvacuationWorkers = Math.max(0, activeEvacuationWorkers - 1);
      }
      if (plan.handoffStage === "bridge") {
        bridgeOutstanding = Math.max(0, bridgeOutstanding - 1);
        bridgeCampaignOutstanding.set(plan.bridgeCampaign,
          Math.max(0, (bridgeCampaignOutstanding.get(plan.bridgeCampaign) || 0) - 1));
        if (queueBridge && !settled) queueBridgePlans();
      }
    };
    const continuePortfolio = () => {
      if (settled) return;
      pumpDirectPlans();
      if (requiredWork.isComplete() && !exactShardsStarted) {
        const requiredSnapshot = requiredWork.snapshot();
        setStatus(
          `Heuristic portfolio complete; starting exact search after ` +
          `${totalVisited.toLocaleString()} states.`,
        );
        appendSearchLog("director", "Required heuristic portfolio completed", {
          requiredWorkers: requiredSnapshot.required,
          opportunisticOutstanding: bridgeOutstanding,
          states: totalVisited.toLocaleString(),
        });
        launchExactShards();
      }
    };
    const abandonWorker = (reason, error = null) => {
      if (workerFinished || settled) return;
      workerFinished = true;
      doneWorkers++;
      if (isRequiredPlan(plan)) requiredWork.finish();
      appendSearchLog(reason === "watchdog" ? "warning" : "error",
        `${plan.label} terminated`, {
          reason,
          visited: (workerProgress.get(worker) || 0).toLocaleString(),
          message: error?.message,
          file: error?.filename,
          line: error?.lineno,
        });
      if (plan.handoffStage === "bridge") {
        const campaign = bridgeCampaignTracker.recordFinished(plan.bridgeCampaign, {
          terminationReason: reason,
          visited: workerProgress.get(worker) || 0,
          workerMs: performance.now() - (workerStarted.get(worker) || performance.now()),
        });
        if (campaign.exhaustedReason && !loggedExhaustedCampaigns.has(plan.bridgeCampaign)) {
          loggedExhaustedCampaigns.add(plan.bridgeCampaign);
          appendSearchLog("bridge", "Bridge campaign circuit breaker opened", {
            campaign: plan.bridgeCampaign,
            reason: campaign.exhaustedReason,
            scheduled: campaign.scheduled,
            incompatible: campaign.incompatible,
            productive: campaign.productive,
            visited: campaign.visited.toLocaleString(),
          });
        }
      }
      releaseWorker();
      if (plan.persistentExact) {
        expectedWorkers++;
        const recovery = {...plan, label: `${plan.label} Recovery`};
        appendSearchLog("director", "Recovering interrupted exact shard", {
          shard: `${plan.exactShard.index + 1}/${plan.exactShard.count}`,
          reason,
        });
        launch(recovery);
        return;
      }
      continuePortfolio();
    };
    const cancelWorker = (reason) => {
      if (workerFinished || settled) return;
      workerFinished = true;
      doneWorkers++;
      if (isRequiredPlan(plan)) requiredWork.finish();
      appendSearchLog("worker", `Retired ${plan.label}`, {
        reason,
        visited: workerProgress.get(worker) || 0,
      });
      releaseWorker({queueBridge: false});
      continuePortfolio();
    };
    workerCancellations.set(worker, cancelWorker);
    const telemetryTimer = setInterval(() => {
      if (!workers.includes(worker)) return;
      const silentSeconds = Math.round(
        (performance.now() - (workerLastMessage.get(worker) || performance.now())) / 1000,
      );
      if (silentSeconds >= 120) {
        abandonWorker("watchdog");
        return;
      }
      appendSearchLog("heartbeat", `${plan.label} remains allocated`, {
          silentFor: `${silentSeconds}s`,
          lastVisited: (workerProgress.get(worker) || 0).toLocaleString(),
        });
    }, 30000);
    searchTelemetryTimers.push(telemetryTimer);
    workerTelemetry.set(worker, telemetryTimer);
    if (plan.side === "forward") {
      workerRecords.set(worker, forwardRecords);
    } else if (plan.side === "reverse") {
      const records = new Map();
      reverseRecordSets.push(records);
      workerRecords.set(worker, records);
    }
    worker.onmessage = ({data}) => {
      workerLastMessage.set(worker, performance.now());
      if (data.type === "records") {
        inspectRecords(data.records, worker);
        return;
      }
      if (data.type === "landmarks") {
        registerLandmarks(data.landmarks || [], worker, plan);
        return;
      }
      if (data.type === "reverse-starts") {
        appendSearchLog("reverse", `${plan.label} assigned solved-side branches`, {
          shard: `${data.shard.index + 1}/${data.shard.count}`,
          anchors: data.assignedRegions,
          productiveAnchors: data.assignedProductiveRegions,
          assignedPulls: data.assignedPullOptions,
          totalPulls: data.totalPullOptions,
          totalAnchors: data.totalRegions,
        });
        return;
      }
      if (data.type === "contour") {
        appendSearchLog("contour", `${plan.label} entered exact bound`, {
          threshold: data.threshold,
          visited: (data.visited || 0).toLocaleString(),
          shard: data.exactShard
            ? `${data.exactShard.index + 1}/${data.exactShard.count}` : undefined,
        });
        return;
      }
      if (data.type === "progress") {
        const previous = workerProgress.get(worker) || 0;
        const delta = data.delta ?? Math.max(0, (data.visited || 0) - previous);
        workerProgress.set(worker, data.visited || previous + delta);
        totalVisited += delta;
        const workerElapsed = Math.max(0.001,
          (performance.now() - (workerStarted.get(worker) || performance.now())) / 1000);
        appendSearchLog("progress", plan.label, {
          local: (data.visited || 0).toLocaleString(),
          total: totalVisited.toLocaleString(),
          rate: `${Math.round((data.visited || 0) / workerElapsed).toLocaleString()}/s`,
          h: Number.isFinite(data.bestEstimate) ? data.bestEstimate : undefined,
          initialH: Number.isFinite(data.initialEstimate) ? data.initialEstimate : undefined,
          pushes: data.bestPushes,
          totalPushes: Number.isFinite(data.bestPushes)
            ? (plan.prefixCost || 0) + data.bestPushes : undefined,
          projected: Number.isFinite(data.bestPushes) && Number.isFinite(data.bestEstimate)
            ? (plan.prefixCost || 0) + data.bestPushes + data.bestEstimate : undefined,
          depth: data.depth,
          threshold: data.threshold,
          frontier: data.frontier,
          retained: data.retained,
          generated: data.generated,
          peakFrontier: data.peakFrontier,
          compactions: data.compactions,
          maxDepth: data.maxDepth,
          nextThreshold: data.nextThreshold,
          thresholdPrunes: data.thresholdPrunes,
          upperBoundPrunes: data.upperBoundPrunes,
          profileMs: data.performance?.totalMs,
          graphMs: data.performance?.graphCompileMs,
          denseMs: data.performance?.denseBuildMs,
          preparedBoardReuses: data.performance?.preparedBoardReuses,
          signatureMs: data.performance?.signatureMs,
          signatureCacheHits: data.performance?.signatureCacheHits,
          heuristicMs: data.performance?.heuristicMs,
          reachabilityMs: data.performance?.reachabilityMs,
          heuristicCacheHits: data.performance?.heuristicCacheHits,
          pushDistanceCacheHits: data.performance?.pushDistanceCacheHits,
          corralPrunes: data.corralPrunes,
          cyclePrunes: data.cyclePrunes,
          transpositionPrunes: data.transpositionPrunes,
          transpositions: data.transpositions,
          transpositionEvictions: data.transpositionEvictions,
          shardRejected: data.shardRejected,
          shardAccepted: data.shardAccepted,
        });
        const phase = plan.side === "direct" ? `${plan.label} - ` : "";
        setStatus(`${workers.length} Ultimate workers searching... ${phase}${totalVisited.toLocaleString()} states`);
        return;
      }
      if (settled) return;
      if (data.type === "done") {
        if (workerFinished) return;
        workerFinished = true;
        const previous = workerProgress.get(worker) || 0;
        const workerElapsedSeconds = Math.max(0,
          (performance.now() - (workerStarted.get(worker) || performance.now())) / 1000);
        totalVisited += Math.max(0, (data.visited || 0) - previous);
        const progress = Number.isFinite(data.bestEstimate)
          ? `, h${data.bestEstimate} @ p${data.bestPushes || 0}`
          : "";
        completedPhases.push(
          `${plan.label}: ${(data.visited || 0).toLocaleString()}${progress}`,
        );
        appendSearchLog("worker", `Finished ${plan.label}`, {
          visited: (data.visited || 0).toLocaleString(),
          elapsed: `${workerElapsedSeconds.toFixed(1)}s`,
          cutoff: Boolean(data.cutoff),
          solved: Boolean(data.path),
          h: Number.isFinite(data.bestEstimate) ? data.bestEstimate : undefined,
          pushes: data.bestPushes,
          totalPushes: Number.isFinite(data.bestPushes)
            ? (plan.prefixCost || 0) + data.bestPushes : undefined,
          projected: Number.isFinite(data.bestPushes) && Number.isFinite(data.bestEstimate)
            ? (plan.prefixCost || 0) + data.bestPushes + data.bestEstimate : undefined,
          generated: data.generated?.toLocaleString(),
          peakFrontier: data.peakFrontier?.toLocaleString(),
          compactions: data.compactions,
          frontier: data.frontier?.toLocaleString(),
          retained: data.retained?.toLocaleString(),
          checkpoints: data.checkpoints?.length,
          maxDepth: data.maxDepth,
          nextThreshold: data.nextThreshold,
          thresholdPrunes: data.thresholdPrunes,
          upperBoundPrunes: data.upperBoundPrunes,
          corralPrunes: data.corralPrunes,
          cyclePrunes: data.cyclePrunes,
          transpositionPrunes: data.transpositionPrunes,
          transpositionEvictions: data.transpositionEvictions,
          maxTranspositions: data.maxTranspositions,
          shardRejected: data.shardRejected,
          shardAccepted: data.shardAccepted,
          profileMs: data.performance?.totalMs,
          graphMs: data.performance?.graphCompileMs,
          denseMs: data.performance?.denseBuildMs,
          preparedBoardReuses: data.performance?.preparedBoardReuses,
          signatureMs: data.performance?.signatureMs,
          signatureCacheHits: data.performance?.signatureCacheHits,
          heuristicMs: data.performance?.heuristicMs,
          reachabilityMs: data.performance?.reachabilityMs,
          heuristicCacheHits: data.performance?.heuristicCacheHits,
          pushDistanceCacheHits: data.performance?.pushDistanceCacheHits,
          reason: data.terminationReason || (data.path ? "solution" : data.cutoff ? "budget" : "exhausted"),
        });
        if (plan.handoffStage === "bridge" && data.path && data.finalState) {
          const reverseRecords = reverseRecordSets[plan.reverseRecordIndex];
          const prefixPath = [...(plan.prefixPath || []), ...data.path];
          const joined = reverseRecords && reconstructCheckpointMeetPath(
            {state: data.finalState}, prefixPath, reverseRecords,
          );
          if (joined && finish(joined, plan.label)) return;
        }
        let bridgeCampaignSnapshot = null;
        if (plan.handoffStage === "bridge") {
          bridgeCampaignSnapshot = bridgeCampaignTracker.recordFinished(plan.bridgeCampaign, {
            terminationReason: data.terminationReason,
            visited: data.visited,
            workerMs: workerElapsedSeconds * 1000,
          });
          if (bridgeCampaignSnapshot.exhaustedReason &&
              !loggedExhaustedCampaigns.has(plan.bridgeCampaign)) {
            loggedExhaustedCampaigns.add(plan.bridgeCampaign);
            appendSearchLog("bridge", "Bridge campaign circuit breaker opened", {
              campaign: plan.bridgeCampaign,
              reason: bridgeCampaignSnapshot.exhaustedReason,
              scheduled: bridgeCampaignSnapshot.scheduled,
              incompatible: bridgeCampaignSnapshot.incompatible,
              productive: bridgeCampaignSnapshot.productive,
              visited: bridgeCampaignSnapshot.visited.toLocaleString(),
            });
          }
        }
        if (plan.handoffStage === "bridge" && data.terminationReason !== "target-incompatible") {
          bridgeCampaignViable.set(plan.bridgeCampaign,
            (bridgeCampaignViable.get(plan.bridgeCampaign) || 0) + 1);
        }
        const continuationDecision = plan.handoffStage === "bridge" && data.checkpoint
          ? SokomindDirectorPolicy.evaluateBridgeContinuation({
            continuation: plan.bridgeContinuation || 0,
            initialEstimate: data.initialEstimate,
            bestEstimate: data.bestEstimate,
            checkpointCost: data.checkpoint.cost,
          }) : null;
        if (plan.handoffStage === "bridge" && data.cutoff && data.checkpoint?.path?.length &&
            continuationDecision?.promote && !exactShardsStarted &&
            bridgeOutstanding < bridgeMaxOutstanding &&
            bridgeCampaignTracker.canSchedule(plan.bridgeCampaign)) {
          const checkpoint = data.checkpoint;
          const remaining = Number.isFinite(plan.upperBound)
            ? plan.upperBound - checkpoint.cost : checkpoint.estimate + 50;
          if (remaining > 0) {
            const continuation = {
              ...plan,
              label: `${plan.label} Continuation ${(plan.bridgeContinuation || 0) + 1}`,
              state: checkpoint.state,
              prefixPath: [...(plan.prefixPath || []), ...checkpoint.path],
              prefixCost: (plan.prefixCost || 0) + checkpoint.cost,
              upperBound: remaining,
              maxVisited: 80000,
              bridgeContinuation: (plan.bridgeContinuation || 0) + 1,
              bridgePairKey: `${plan.bridgePairKey}:c${(plan.bridgeContinuation || 0) + 1}`,
              queuePriority: 5,
            };
            bridgeCampaignTracker.recordScheduled(plan.bridgeCampaign);
            bridgeOutstanding++;
            bridgeCampaignOutstanding.set(plan.bridgeCampaign,
              (bridgeCampaignOutstanding.get(plan.bridgeCampaign) || 0) + 1);
            enqueueDirectPlans([continuation], {priority: 5});
            appendSearchLog("bridge", "Promising bridge checkpoint promoted", {
              source: plan.label, h: checkpoint.estimate,
              localPushes: checkpoint.cost, continuation: continuation.bridgeContinuation,
              reason: continuationDecision.reason,
              improvement: continuationDecision.improvement,
              efficiency: continuationDecision.efficiency.toFixed(2),
            });
          }
        }
        const completePath = plan.handoffStage !== "bridge" && plan.prefixPath && data.path
          ? [...plan.prefixPath, ...data.path]
          : plan.handoffStage !== "bridge" ? data.path : null;
        if (plan.side === "direct" && completePath && finish(completePath, plan.label)) return;
        if (plan.side === "direct" && plan.handoffStage !== "bridge" &&
            registerCheckpoints(plan, data)) return;
        if (plan.side === "direct" && plan.handoffStage !== "bridge") {
          queueCheckpointWorkers(plan, data);
        }
        const exhaustedExactShard = plan.persistentExact && !data.path && !data.cutoff;
        if (exhaustedExactShard) exactShardsExhausted++;
        const exactRoundComplete = exhaustedExactShard &&
          exactShardsExhausted === exactRoundShardCount;
        const provedUnsolvable = exactRoundComplete && !Number.isFinite(effectiveUpperBound);
        if (exactRoundComplete && Number.isFinite(effectiveUpperBound)) {
          discardedExactIncumbent = true;
          exactShardsStarted = false;
          appendSearchLog("director", "Stored incumbent bound did not replay; removing it from exact search", {
            bound: effectiveUpperBound,
          });
        }
        doneWorkers++;
        if (isRequiredPlan(plan)) requiredWork.finish();
        releaseWorker();
        if (provedUnsolvable) {
          settled = true;
          setControlsBusy(false);
          setStatus(
            `Exact IDA* proved that this puzzle has no solution ` +
            `(${totalVisited.toLocaleString()} states).`,
          );
          appendSearchLog("proof", "Exact search exhausted the reachable push space", {
            states: totalVisited.toLocaleString(),
          });
          return;
        }
        continuePortfolio();
      }
    };
    worker.onerror = error => abandonWorker("worker-error", error);
    const planState = plan.state || serializeState(state);
    worker.postMessage({
      ...plan,
      state: {...planState, preparedBoard: analysis.preparedBoard},
      upperBound: effectiveUpperBound,
    });
  };

  launch({
    mode: "bidir-forward",
    side: "forward",
    label: "Forward Push Search",
    maxVisited: sideVisitedLimit,
    frontierLimit: 40000,
  });
  for (let index = 0; index < reverseWorkers; index++) {
    launch({
      mode: "bidir-reverse",
      side: "reverse",
      label: `Reverse Branch Shard ${index + 1}/${reverseWorkers}`,
      landmarkGeneration: `initial-${index + 1}`,
      maxVisited: sideVisitedLimit,
      frontierLimit: 40000,
      reverseShard: {index, count: reverseWorkers},
    });
  }
  pumpDirectPlans();
}
function startSolver(purpose) {
  if ($("algorithm").value === "ultimate-bidirectional") {
    startBidirectionalSolver(purpose);
    return;
  }
  stop(false); setControlsBusy(true); resetSearchLog();
  setStatus(`${$("algorithm").selectedOptions[0].text} is searching...`);
  const plans = solverPlans($("algorithm").value);
  const maxWorkers = Math.max(1, Math.min(plans.length, navigator.hardwareConcurrency || 2));
  const queue = plans.slice(), active = new Map(), finished = [];
  let settled = false, totalVisited = 0;
  appendSearchLog("director", "Started search portfolio", {
    algorithm: $("algorithm").value, workers: maxWorkers, plans: plans.length,
  });
  const launchNext = () => {
    if (settled || !queue.length || active.size >= maxWorkers) return;
    const plan = queue.shift(), worker = new Worker(SOLVER_WORKER_URL);
    workers.push(worker); active.set(worker, plan);
    const launchedAt = performance.now();
    appendSearchLog("worker", `Started ${plan.label}`, {
      algorithm: plan.algorithm, budget: plan.maxVisited?.toLocaleString(),
      width: plan.beamWidth,
    });
    worker.onmessage = ({data}) => {
      if (settled) return;
      if (data.type === "progress") {
        const seconds = Math.max(.001, (performance.now() - launchedAt) / 1000);
        appendSearchLog("progress", plan.label, {
          local: data.visited.toLocaleString(),
          rate: `${Math.round(data.visited / seconds).toLocaleString()}/s`,
          h: Number.isFinite(data.bestEstimate) ? data.bestEstimate : undefined,
          depth: data.depth, threshold: data.threshold, frontier: data.frontier,
          graphMs: data.performance?.graphCompileMs,
          denseMs: data.performance?.denseBuildMs,
          preparedBoardReuses: data.performance?.preparedBoardReuses,
          signatureMs: data.performance?.signatureMs,
          heuristicMs: data.performance?.heuristicMs,
          reachabilityMs: data.performance?.reachabilityMs,
        });
        setStatus(`${active.size} worker${active.size === 1 ? "" : "s"} searching... ${plan.label}: ${data.visited.toLocaleString()} states`);
        return;
      }
      worker.terminate();
      workers = workers.filter(item => item !== worker);
      active.delete(worker);
      totalVisited += data.visited || 0;
      appendSearchLog("worker", `Finished ${plan.label}`, {
        visited: (data.visited || 0).toLocaleString(), solved: Boolean(data.path),
        cutoff: Boolean(data.cutoff),
        h: Number.isFinite(data.bestEstimate) ? data.bestEstimate : undefined,
        profileMs: data.performance?.totalMs,
        graphMs: data.performance?.graphCompileMs,
        denseMs: data.performance?.denseBuildMs,
        preparedBoardReuses: data.performance?.preparedBoardReuses,
        signatureMs: data.performance?.signatureMs,
        signatureCacheHits: data.performance?.signatureCacheHits,
        heuristicMs: data.performance?.heuristicMs,
        reachabilityMs: data.performance?.reachabilityMs,
      });
      if (data.path) {
        const path = validatePathToGoal(data.path);
        if (path !== null) {
          settled = true;
          appendSearchLog("solution", `${plan.label} produced a replay-validated solution`, {
            moves: path.length, states: totalVisited.toLocaleString(),
          });
          workers.forEach(item => item.terminate()); workers = [];
          setControlsBusy(false);
          if (purpose === "hint") {
            setStatus(path.length
              ? `Hint: ${path[0]} - ${path.length} moves remain (${plan.label})`
              : "This puzzle is already solved.");
          } else {
            setStatus(`Found ${path.length} moves with ${plan.label} after ${totalVisited.toLocaleString()} states.`);
            animation = path; animate();
          }
          return;
        }
      }
      finished.push(data);
      if (!queue.length && active.size === 0) {
        setControlsBusy(false);
        setStatus(`No solution found (${totalVisited.toLocaleString()} states across ${finished.length} worker${finished.length === 1 ? "" : "s"}).`);
        appendSearchLog("result", "Portfolio ended without a solution", {
          states: totalVisited.toLocaleString(), workers: finished.length,
        });
        return;
      }
      launchNext();
    };
    worker.onerror = () => {
      if (settled) return;
      worker.terminate();
      workers = workers.filter(item => item !== worker);
      active.delete(worker);
      appendSearchLog("error", `${plan.label} worker failed`);
      finished.push({visited: 0});
      launchNext();
      if (!queue.length && active.size === 0) {
        setControlsBusy(false);
        setStatus("Solver worker failed.");
      }
    };
    worker.postMessage({state: serializeState(state), upperBound: planUpperBound(plan), ...plan});
    launchNext();
  };
  launchNext();
}
function animate() {
  if (!animation.length || isGoal(state)) return;
  tryMove(animation.shift(), true);
  if (animation.length) timer = setTimeout(animate, 105);
}

$("solve").onclick = () => startSolver("solve");
$("home-button").onclick = showHome;
$("start-game").onclick = hideHome;
$("hint").onclick = () => startSolver("hint");
$("stop").onclick = () => stop();
$("undo").onclick = undo; $("reset").onclick = reset;
$("copy-moves").onclick = async () => {
  try {
    await navigator.clipboard.writeText(moveHistoryText());
    setStatus(`Copied ${moves.toLocaleString()} moves.`);
  } catch (_error) {
    setStatus("Could not copy move history.");
  }
};
$("copy-search-log").onclick = async () => {
  try {
    await navigator.clipboard.writeText(searchLogText());
    setStatus(`Copied ${searchLog.length.toLocaleString()} search log entries.`);
  } catch (_error) {
    setStatus("Could not copy search log.");
  }
};
$("copy-search-json").onclick = async () => {
  try {
    await navigator.clipboard.writeText(searchLogJsonLines());
    setStatus(`Copied ${searchLog.length.toLocaleString()} structured JSONL events.`);
  } catch (_error) {
    setStatus("Could not copy structured search log.");
  }
};
$("replay").onclick = () => { $("complete-dialog").close(); reset(); };
$("next-level").onclick = () => {
  $("complete-dialog").close();
  const keys = Object.keys(LEVELS); loadLevel(keys[keys.indexOf(levelKey) + 1]);
};
$("close-dialog").onclick = () => $("complete-dialog").close();
document.querySelectorAll(".touch-button").forEach(button => {
  const move = button.dataset.move;
  button.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    if (!$("home-screen").classList.contains("hidden") || $("complete-dialog").open) return;
    stop(false); tryMove(move); $("board").focus();
  });
});
document.addEventListener("keydown", (event) => {
  if (!$("home-screen").classList.contains("hidden") ||
      $("complete-dialog").open ||
      SokomindKeyboard.shouldIgnoreGameShortcut(event.target)) return;
  const direction = KEYS[event.key];
  if (direction) { event.preventDefault(); stop(false); tryMove(direction); }
  else if (event.key === "Backspace" || event.key.toLowerCase() === "u") undo();
  else if (event.key.toLowerCase() === "r") reset();
});
window.addEventListener("resize", () => { fitBoardToScreen(); });
loadLevel(levelKey);

