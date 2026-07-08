const LEVELS = {
  "ultra-tiny": ["OOOOO", "O R O", "O A O", "O a O", "OOOOO"],
  tiny: ["OOOOOO", "O R  O", "O XO O", "OO A O", "OSa  O", "OOOOOO"],
  medium: ["OOOOOOO", "Oa   bO", "O AXB O", "O XRX O", "OSCXDSO", "OcS SdO", "OOOOOOO"],
  large: ["OOOOOOOOOO", "OOOOOOOSSO", "OOOOO  abO", "OOOOO XSSO", "OOOOOO  OO",
    "OR     OOO", "OO A X X O", "OO BXO O O", "OO   O   O", "OOOOOOOOOO"],
  huge: ["OOOOOOOOOOOOOOO", "OaSS   S   SSbO", "OSCS       SDSO", "OX X       X XO",
    "O             O", "OOOO   X   OOOO", "O      O      O", "O G hOOOOOH g O",
    "O      O      O", "O             O", "O     X X     O", "OOOOOOOROOOOOOO",
    "O B X X X X A O", "O Sc       dS O", "OOOOOOOOOOOOOOO"],
};
const DIRS = {Up: [-1, 0], Down: [1, 0], Left: [0, -1], Right: [0, 1]};
const KEYS = {ArrowUp: "Up", ArrowDown: "Down", ArrowLeft: "Left", ArrowRight: "Right",
  w: "Up", W: "Up", s: "Down", S: "Down", a: "Left", A: "Left", d: "Right", D: "Right"};
const $ = (id) => document.getElementById(id);

let levelKey = "ultra-tiny", state, initialState, history = [], moves = 0;
let worker = null, animation = [], timer = null, solvedShown = false;
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
    copy.innerHTML = `<strong>${title(key)}</strong><small>${rows[0].length} × ${rows.length}</small>`;
    button.append(thumb, copy); button.onclick = () => loadLevel(key);
    return button;
  }));
}
function render() {
  $("level-title").textContent = title(levelKey);
  $("move-count").textContent = moves;
  const board = $("board"), rows = state.board.rows;
  board.style.gridTemplateColumns = `repeat(${rows[0].length}, 1fr)`;
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
function loadLevel(key) {
  stop(); levelKey = key; state = parse(LEVELS[key]); initialState = cloneState(state);
  history = []; moves = 0; solvedShown = false; resetTimer();
  setStatus("Use arrow keys or WASD to play."); renderLevels(); render();
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
  stop(false); freezeTimer(); setStatus(`Solved in ${moves} moves!`);
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
  if (worker) worker.terminate(); worker = null;
  animation = []; clearTimeout(timer); timer = null;
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
function startSolver(purpose) {
  stop(false); setStatus(`${$("algorithm").selectedOptions[0].text} is searching…`);
  worker = new Worker("solver-worker.js");
  worker.onmessage = ({data}) => {
    if (data.type === "progress") {
      setStatus(`Searching… ${data.visited.toLocaleString()} states`);
      return;
    }
    worker.terminate(); worker = null;
    if (!data.path) { setStatus(`No solution found (${data.visited.toLocaleString()} states).`); return; }
    if (purpose === "hint") {
      setStatus(`Hint: ${data.path[0]} · ${data.path.length} moves remain`);
    } else {
      setStatus(`Found ${data.path.length} moves after ${data.visited.toLocaleString()} states.`);
      animation = data.path; animate();
    }
  };
  worker.postMessage({state: serializeState(state), algorithm: $("algorithm").value});
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
document.addEventListener("keydown", (event) => {
  if (!$("home-screen").classList.contains("hidden") ||
      $("complete-dialog").open || event.target.matches("select, button")) return;
  const direction = KEYS[event.key];
  if (direction) { event.preventDefault(); stop(false); tryMove(direction); }
  else if (event.key === "Backspace" || event.key.toLowerCase() === "u") undo();
  else if (event.key.toLowerCase() === "r") reset();
});
loadLevel(levelKey);
