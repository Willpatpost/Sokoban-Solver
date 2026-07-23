const LEVELS = SokomindLevels.LEVELS;
const {
  DIRS,
  positionKey: pos,
  parseRows: parse,
  cloneState,
  isGoal,
  moveState,
  isPushMove,
  serializeState,
} = SokomindGameState;
const {
  text: formatSearchLogText,
  jsonLines: formatSearchLogJsonLines,
  structuredStats: structuredSearchStats,
  formatTime,
  shortStateId,
} = SokomindSearchLog;
const CODE_MOVE = {U: "Up", D: "Down", L: "Left", R: "Right"};
const SOLVER_BUILD = "2026-07-22.24";
const SOLVER_WORKER_URL = `solver-worker.js?build=${SOLVER_BUILD}`;
const PUSH_BOUNDS_KEY = "sokomind-push-bounds-v1";
const KEYS = {ArrowUp: "Up", ArrowDown: "Down", ArrowLeft: "Left", ArrowRight: "Right",
  w: "Up", W: "Up", s: "Down", S: "Down", a: "Left", A: "Left", d: "Right", D: "Right"};
const $ = (id) => document.getElementById(id);

$("solver-build").textContent = SOLVER_BUILD;

let levelKey = "ultra-tiny", state, initialState, history = [], moveHistory = [], moves = 0;
let workers = [], animation = [], timer = null, solvedShown = false;
let startedAt = null, elapsed = 0, clock = null;
let pushBounds = {};
let searchLog = [], searchStartedAt = null;
let searchRunId = null, searchLogSequence = 0;
let searchTelemetryTimers = [];

const searchLogText = (entries = searchLog) => formatSearchLogText(entries);
const searchLogJsonLines = (entries = searchLog) => formatSearchLogJsonLines(entries);

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

function moveHistoryText() {
  return moveHistory.map(({direction, pushed}, index) => (
    `${index + 1}. ${direction}${pushed ? " (push)" : ""}`
  )).join("\n");
}
function renderMoveHistory() {
  $("move-history-count").textContent = moves.toLocaleString();
  $("move-history-text").value = moveHistoryText();
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

