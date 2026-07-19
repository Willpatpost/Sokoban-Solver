const LEVELS = {
  "ultra-tiny": ["OOOOO", "O R O", "O A O", "O a O", "OOOOO"],
  tiny: ["OOOOOO", "O R  O", "O XO O", "OO A O", "OSa  O", "OOOOOO"],
  medium: ["OOOOOOO", "Oa   bO", "O AXB O", "O XRX O", "OSCXDSO", "OcS SdO", "OOOOOOO"],
  large: ["OOOOOOOOOO", "OOOOOOOSSO", "OOOOO  abO", "OOOOO XSSO", "OOOOOO  OO",
    "OR     OOO", "OO A X X O", "OO BXO O O", "OO   O   O", "OOOOOOOOOO"],
  huge: ["OOOOOOOOOOOOOOO", "OaSS   S   SSbO", "OSCS  OOO  SDSO", "OX X  OOO  X XO",
    "O     OOO     O", "OOOO   X   OOOO", "O      O      O", "O G hOOOOOH g O",
    "O      O      O", "OOO         OOO", "OOO   X X   OOO", "OOOOOOOROOOOOOO",
    "O B X X X X A O", "O Sc       dS O", "OOOOOOOOOOOOOOO"],
};
const DIRS = {Up: [-1, 0], Down: [1, 0], Left: [0, -1], Right: [0, 1]};
const CODE_MOVE = {U: "Up", D: "Down", L: "Left", R: "Right"};
const SOLVER_BUILD = "2026-07-19.1";
const SOLVER_WORKER_URL = `solver-worker.js?build=${SOLVER_BUILD}`;
const VERIFIED_PUSH_BOUNDS = {huge: 250};
const PUSH_BOUNDS_KEY = "sokomind-push-bounds-v1";
const KEYS = {ArrowUp: "Up", ArrowDown: "Down", ArrowLeft: "Left", ArrowRight: "Right",
  w: "Up", W: "Up", s: "Down", S: "Down", a: "Left", A: "Left", d: "Right", D: "Right"};
const $ = (id) => document.getElementById(id);

let levelKey = "ultra-tiny", state, initialState, history = [], moveHistory = [], moves = 0;
let workers = [], animation = [], timer = null, solvedShown = false;
let startedAt = null, elapsed = 0, clock = null;
let pushBounds = {...VERIFIED_PUSH_BOUNDS};

try {
  const storedBounds = JSON.parse(localStorage.getItem(PUSH_BOUNDS_KEY) || "{}");
  for (const [key, value] of Object.entries(storedBounds)) {
    if (!Number.isInteger(value) || value <= 0) continue;
    pushBounds[key] = Math.min(pushBounds[key] ?? Infinity, value);
  }
} catch (_error) {
  // Storage can be unavailable in private browsing; verified bounds still work.
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
  renderMoveHistory();
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
function stop(message = true) {
  workers.forEach(worker => worker.terminate()); workers = [];
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
function verifiedSolutionPath() {
  if (history.length) return null;
  const encoded = globalThis.SokomindVerifiedSolutions?.[levelKey];
  if (!encoded) return null;
  const path = [...encoded].map(code => CODE_MOVE[code]);
  if (path.some(move => !move)) return null;
  return validatePathToGoal(path);
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
  const verified = verifiedSolutionPath();
  if (verified !== null) {
    setControlsBusy(false);
    const pushes = pushBounds[levelKey];
    const source = `verified ${pushes}-push incumbent`;
    if (purpose === "hint") {
      setStatus(verified.length
        ? `Hint: ${verified[0]} - ${verified.length} moves remain (${source})`
        : "This puzzle is already solved.");
    } else {
      setStatus(`Loaded ${source}: ${verified.length} replayable moves.`);
      animation = verified;
      animate();
    }
    return;
  }
  setStatus(`Ultimate Bidirectional ${SOLVER_BUILD} is searching...`);
  const hardware = navigator.hardwareConcurrency || 2;
  const searchScale = state.boxes.size * state.board.floor.size;
  const reverseLimit = searchScale >= 1200 ? 1 : searchScale >= 500 ? 2 : 3;
  const sideVisitedLimit = searchScale >= 1200 ? 100000 : searchScale >= 500 ? 250000 : undefined;
  const reverseWorkers = Math.max(1, Math.min(hardware - 1, reverseLimit));
  const beamProfiles = [
    {beamProfile: "balanced", weight: 3, diversity: 1.75,
      topologyWeight: 0.8, goalPackingWeight: 0.8},
    {beamProfile: "detour", weight: 2, diversity: 2.5,
      topologyWeight: 1.4, goalPackingWeight: 1.1},
    {beamProfile: "milestone", weight: 2.3, diversity: 2,
      topologyWeight: 2, goalPackingWeight: 1.3},
  ];
  const beamAttemptCount = 2;
  const beamPlans = Array.from({length: beamAttemptCount}, (_item, index) => ({
    algorithm: "push-beam",
    side: "direct",
    label: `Beam Restart ${index + 1}/${beamAttemptCount}`,
    beamWidth: searchScale >= 1200 ? 300 : 1000,
    maxDepth: 600,
    maxVisited: searchScale >= 1200 ? 110000 : 250000,
    transpositionLimit: searchScale >= 1200 ? 18000 : 45000,
    seed: 29 + index * 104729,
    ...beamProfiles[index % beamProfiles.length],
  }));
  const macroProfiles = [
    {beamProfile: "milestone", weight: 2, topologyWeight: 1.4,
      goalPackingWeight: 1.2, diversity: 2, seed: 29, beamWidth: 40,
      sequenceMacroResults: 6},
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
  const macroPlans = searchScale >= 1200 ? macroProfiles.map((settings, index) => ({
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
    ...settings,
  })) : [];
  const dfsProfiles = [
    {dfsProfile: "setup", discrepancyLimit: 0, maxVisited: 20000},
    {dfsProfile: "setup", discrepancyLimit: 2, maxVisited: 80000},
  ];
  const dfsPlans = searchScale >= 1200 ? dfsProfiles.map((settings, index) => ({
    algorithm: "bounded-push-dfs",
    side: "direct",
    label: `Discrepancy DFS ${index + 1}/${dfsProfiles.length}`,
    maxDepth: 600,
    transpositionLimit: 60000,
    diversity: 1.5,
    seed: 313337 + index * 104729,
    ...settings,
  })) : [];
  const directPlans = [...beamPlans, ...macroPlans, ...dfsPlans];
  let nextDirectPlan = 0;
  const forwardRecords = new Map(), reverseRecordSets = [];
  const workerSides = new Map(), workerRecords = new Map(), workerProgress = new Map();
  const expectedWorkers = reverseWorkers + 1 + directPlans.length;
  let settled = false, totalVisited = 0, doneWorkers = 0;
  const completedPhases = [];

  const finish = (path, strategy = "Bidirectional") => {
    const validated = validatePathToGoal(path);
    if (validated === null) return false;
    settled = true;
    workers.forEach(worker => worker.terminate()); workers = [];
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
    return finish(path);
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
    }
    for (const [meetKey, reverseRecords] of meetings) {
      if (requestSolve(meetKey, reverseRecords)) return;
    }
  };

  const launch = (plan) => {
    const worker = new Worker(SOLVER_WORKER_URL);
    workers.push(worker);
    workerSides.set(worker, plan.side);
    if (plan.side === "forward") {
      workerRecords.set(worker, forwardRecords);
    } else if (plan.side === "reverse") {
      const records = new Map();
      reverseRecordSets.push(records);
      workerRecords.set(worker, records);
    }
    worker.onmessage = ({data}) => {
      if (data.type === "records") {
        inspectRecords(data.records, worker);
        return;
      }
      if (data.type === "progress") {
        const previous = workerProgress.get(worker) || 0;
        const delta = data.delta ?? Math.max(0, (data.visited || 0) - previous);
        workerProgress.set(worker, data.visited || previous + delta);
        totalVisited += delta;
        const phase = plan.side === "direct" ? `${plan.label} - ` : "";
        setStatus(`${workers.length} Ultimate workers searching... ${phase}${totalVisited.toLocaleString()} states`);
        return;
      }
      if (settled) return;
      if (data.type === "done") {
        const previous = workerProgress.get(worker) || 0;
        totalVisited += Math.max(0, (data.visited || 0) - previous);
        const progress = Number.isFinite(data.bestEstimate)
          ? `, h${data.bestEstimate} @ p${data.bestPushes || 0}`
          : "";
        completedPhases.push(
          `${plan.label}: ${(data.visited || 0).toLocaleString()}${progress}`,
        );
        if (plan.side === "direct" && data.path && finish(data.path, plan.label)) return;
        doneWorkers++;
        worker.terminate();
        workers = workers.filter(item => item !== worker);
        workerSides.delete(worker);
        workerProgress.delete(worker);
        workerRecords.delete(worker);
        if (plan.side === "direct" && nextDirectPlan < directPlans.length) {
          launch(directPlans[nextDirectPlan++]);
        }
        if (!settled && doneWorkers === expectedWorkers) {
          setControlsBusy(false);
          setStatus(
            `No solution found by ${SOLVER_BUILD} (${totalVisited.toLocaleString()} states). ` +
            completedPhases.join("; "),
          );
        }
      }
    };
    worker.onerror = () => {
      if (settled) return;
      doneWorkers++;
      worker.terminate();
      workers = workers.filter(item => item !== worker);
      workerSides.delete(worker);
      workerProgress.delete(worker);
      workerRecords.delete(worker);
      if (plan.side === "direct" && nextDirectPlan < directPlans.length) {
        launch(directPlans[nextDirectPlan++]);
      }
      if (doneWorkers === expectedWorkers) {
        setControlsBusy(false);
        setStatus("Bidirectional worker failed.");
      }
    };
    worker.postMessage({state: serializeState(state), upperBound: planUpperBound(plan), ...plan});
  };

  launch({
    mode: "bidir-forward",
    side: "forward",
    label: "Forward Push Search",
    maxVisited: sideVisitedLimit,
  });
  for (let index = 0; index < reverseWorkers; index++) {
    launch({
      mode: "bidir-reverse",
      side: "reverse",
      label: `Reverse Unsolver ${index + 1}`,
      maxVisited: sideVisitedLimit,
      reverseShard: {index, count: reverseWorkers},
    });
  }
  launch(directPlans[nextDirectPlan++]);
}
function startSolver(purpose) {
  if ($("algorithm").value === "ultimate-bidirectional") {
    startBidirectionalSolver(purpose);
    return;
  }
  stop(false); setControlsBusy(true); setStatus(`${$("algorithm").selectedOptions[0].text} is searching...`);
  const plans = solverPlans($("algorithm").value);
  const maxWorkers = Math.max(1, Math.min(plans.length, navigator.hardwareConcurrency || 2));
  const queue = plans.slice(), active = new Map(), finished = [];
  let settled = false, totalVisited = 0;
  const launchNext = () => {
    if (settled || !queue.length || active.size >= maxWorkers) return;
    const plan = queue.shift(), worker = new Worker(SOLVER_WORKER_URL);
    workers.push(worker); active.set(worker, plan);
    worker.onmessage = ({data}) => {
      if (settled) return;
      if (data.type === "progress") {
        setStatus(`${active.size} worker${active.size === 1 ? "" : "s"} searching... ${plan.label}: ${data.visited.toLocaleString()} states`);
        return;
      }
      worker.terminate();
      workers = workers.filter(item => item !== worker);
      active.delete(worker);
      totalVisited += data.visited || 0;
      if (data.path) {
        const path = validatePathToGoal(data.path);
        if (path !== null) {
          settled = true;
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
        return;
      }
      launchNext();
    };
    worker.onerror = () => {
      if (settled) return;
      worker.terminate();
      workers = workers.filter(item => item !== worker);
      active.delete(worker);
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
      $("complete-dialog").open || event.target.matches("select, button")) return;
  const direction = KEYS[event.key];
  if (direction) { event.preventDefault(); stop(false); tryMove(direction); }
  else if (event.key === "Backspace" || event.key.toLowerCase() === "u") undo();
  else if (event.key.toLowerCase() === "r") reset();
});
window.addEventListener("resize", () => { fitBoardToScreen(); });
loadLevel(levelKey);

