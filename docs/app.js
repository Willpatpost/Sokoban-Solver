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
const KEYS = {ArrowUp: "Up", ArrowDown: "Down", ArrowLeft: "Left", ArrowRight: "Right",
  w: "Up", W: "Up", s: "Down", S: "Down", a: "Left", A: "Left", d: "Right", D: "Right"};
const $ = (id) => document.getElementById(id);

let levelKey = "ultra-tiny", state, initialState, history = [], moves = 0;
let workers = [], animation = [], timer = null, solvedShown = false;
let startedAt = null, elapsed = 0, clock = null;

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
  history = []; moves = 0; solvedShown = false; resetTimer();
  setStatus("Use arrow keys or WASD to play."); renderLevels(); render();
}
function setControlsBusy(active) {
  $("solve").disabled = active;
  $("hint").disabled = active;
  $("algorithm").disabled = active;
}
function tryMove(direction, fromSolver = false) {
  const next = moveState(state, direction);
  if (!next) { if (!fromSolver) setStatus(`${direction} is blocked.`); return false; }
  startTimer();
  history.push(cloneState(state)); state = next; moves++; render();
  if (isGoal(state)) complete(); else if (!fromSolver) setStatus("Playing");
  return true;
}
function complete() {
  stop(false); setControlsBusy(false); freezeTimer(); setStatus(`Solved in ${moves} moves!`);
  if (solvedShown) return; solvedShown = true;
  $("complete-level").textContent = title(levelKey);
  $("complete-moves").textContent = moves;
  const keys = Object.keys(LEVELS), hasNext = keys.indexOf(levelKey) < keys.length - 1;
  $("next-level").hidden = !hasNext; $("complete-dialog").showModal();
}
function setStatus(text) { $("status").textContent = text; }
function undo() {
  stop(); if (!history.length) return;
  state = history.pop(); moves--; solvedShown = false; render(); setStatus("Undid one move.");
}
function reset() {
  stop(); state = cloneState(initialState); history = []; moves = 0; solvedShown = false;
  resetTimer();
  render(); setStatus("Level reset.");
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
function trimPathToGoal(path) {
  const replay = cloneState(state), trimmed = [];
  for (const move of path) {
    const next = moveState(replay, move);
    if (!next) break;
    replay.robot = next.robot; replay.boxes = next.boxes;
    trimmed.push(move);
    if (isGoal(replay)) break;
  }
  return trimmed;
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
function reconstructMeetPath(meetKey, forwardSeen, reverseSeen) {
  const forwardSegments = [];
  let current = forwardSeen.get(meetKey);
  if (!current || !reverseSeen.has(meetKey)) return null;
  while (current?.parent) {
    forwardSegments.unshift(...current.segment);
    current = forwardSeen.get(current.parent);
    if (!current) return null;
  }

  const reverseSegments = [];
  current = reverseSeen.get(meetKey);
  while (current?.parent) {
    reverseSegments.push(...current.segment);
    current = reverseSeen.get(current.parent);
    if (!current) return null;
  }

  const forward = forwardSeen.get(meetKey), reverse = reverseSeen.get(meetKey);
  const bridge = walkBetween(state.board, boxesFromMeetKey(forward.key), forward.robot, reverse.robot);
  if (!bridge) return null;
  return [...forwardSegments, ...bridge, ...reverseSegments];
}
function solverPlans(algorithm) {
  if (!["ultimate", "portfolio", "fast"].includes(algorithm)) {
    return [{algorithm, label: $("algorithm").selectedOptions[0].text}];
  }
  return [
    {algorithm: "push-greedy", label: "Push Greedy"},
    {algorithm: "weighted-push-astar", label: "Weighted Push A*"},
    {algorithm: "push-astar", label: "Push A*"},
  ];
}
function startBidirectionalSolver(purpose) {
  stop(false); setControlsBusy(true); setStatus("Ultimate Bidirectional is searching...");
  const hardware = navigator.hardwareConcurrency || 2;
  const reverseWorkers = Math.max(1, Math.min(hardware - 1, 3));
  const forwardRecords = new Map(), reverseRecords = new Map();
  const workerSides = new Map();
  let settled = false, totalVisited = 0, doneWorkers = 0;

  const finish = (path) => {
    settled = true;
    workers.forEach(worker => worker.terminate()); workers = [];
    setControlsBusy(false);
    const trimmed = trimPathToGoal(path);
    if (purpose === "hint") {
      setStatus(`Hint: ${trimmed[0]} - ${trimmed.length} moves remain (Bidirectional)`);
    } else {
      setStatus(`Met in the middle: ${trimmed.length} moves after ${totalVisited.toLocaleString()} states.`);
      animation = trimmed; animate();
    }
  };

  const requestSolve = (meetKey) => {
    if (settled) return false;
    const path = reconstructMeetPath(meetKey, forwardRecords, reverseRecords);
    if (!path) return false;
    finish(path);
    return true;
  };

  const inspectRecords = (records, worker) => {
    const side = workerSides.get(worker);
    const recordMap = side === "forward" ? forwardRecords : reverseRecords;
    const otherMap = side === "forward" ? reverseRecords : forwardRecords;
    const meetings = [];
    for (const record of records) {
      recordMap.set(record.h, record);
      if (otherMap.has(record.h) && otherMap.get(record.h).key === record.key) {
        meetings.push(record.h);
      }
    }
    for (const meetKey of meetings) {
      if (requestSolve(meetKey)) return;
    }
  };

  const launch = (plan) => {
    const worker = new Worker("solver-worker.js");
    workers.push(worker);
    workerSides.set(worker, plan.side);
    worker.onmessage = ({data}) => {
      if (data.type === "records") {
        inspectRecords(data.records, worker);
        return;
      }
      if (data.type === "progress") {
        totalVisited += data.delta || 0;
        setStatus(`${workers.length} bidirectional workers searching... ${totalVisited.toLocaleString()} states`);
        return;
      }
      if (settled) return;
      if (data.type === "done") {
        doneWorkers++;
        worker.terminate();
        workers = workers.filter(item => item !== worker);
        workerSides.delete(worker);
        if (!settled && doneWorkers === reverseWorkers + 1) {
          setControlsBusy(false);
          setStatus(`No bidirectional meeting found (${totalVisited.toLocaleString()} states).`);
        }
      }
    };
    worker.onerror = () => {
      if (settled) return;
      doneWorkers++;
      worker.terminate();
      workers = workers.filter(item => item !== worker);
      workerSides.delete(worker);
      if (doneWorkers === reverseWorkers + 1) {
        setControlsBusy(false);
        setStatus("Bidirectional worker failed.");
      }
    };
    worker.postMessage({state: serializeState(state), ...plan});
  };

  launch({mode: "bidir-forward", side: "forward", label: "Forward Push Search"});
  for (let index = 0; index < reverseWorkers; index++) {
    launch({
      mode: "bidir-reverse",
      side: "reverse",
      label: `Reverse Unsolver ${index + 1}`,
      reverseShard: {index, count: reverseWorkers},
    });
  }
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
    const plan = queue.shift(), worker = new Worker("solver-worker.js");
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
        settled = true;
        workers.forEach(item => item.terminate()); workers = [];
        setControlsBusy(false);
        const path = trimPathToGoal(data.path);
        if (purpose === "hint") {
          setStatus(`Hint: ${path[0]} - ${path.length} moves remain (${plan.label})`);
        } else {
          setStatus(`Found ${path.length} moves with ${plan.label} after ${totalVisited.toLocaleString()} states.`);
          animation = path; animate();
        }
        return;
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
    worker.postMessage({state: serializeState(state), ...plan});
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

