(function attachGameState(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.SokomindGameState = api;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  "use strict";

  const DIRS = Object.freeze({
    Up: Object.freeze([-1, 0]),
    Down: Object.freeze([1, 0]),
    Left: Object.freeze([0, -1]),
    Right: Object.freeze([0, 1]),
  });

  const positionKey = (y, x) => `${y},${x}`;

  function parseRows(rows) {
    const board = {rows, walls: new Set(), goals: new Map(), floor: new Set()};
    const boxes = new Map();
    let robot;
    rows.forEach((row, y) => [...row].forEach((cell, x) => {
      const position = positionKey(y, x);
      if (cell === "O") board.walls.add(position);
      else board.floor.add(position);
      if (cell === "R") robot = [y, x];
      else if (cell === "X" || (/[A-Z]/.test(cell) && !"ORS".includes(cell))) {
        boxes.set(position, cell);
      } else if (cell === "S") board.goals.set(position, "X");
      else if (/[a-z]/.test(cell)) board.goals.set(position, cell.toUpperCase());
    }));
    return {board, robot, boxes};
  }

  const cloneState = state => ({
    board: state.board,
    robot: [...state.robot],
    boxes: new Map(state.boxes),
  });

  function isGoal(state) {
    return [...state.boxes].every(
      ([position, label]) => state.board.goals.get(position) === label,
    );
  }

  function moveState(state, direction) {
    const delta = DIRS[direction];
    if (!delta) return null;
    const [dy, dx] = delta;
    const [y, x] = state.robot;
    const next = positionKey(y + dy, x + dx);
    if (!state.board.floor.has(next)) return null;
    const result = cloneState(state);
    if (state.boxes.has(next)) {
      const beyond = positionKey(y + 2 * dy, x + 2 * dx);
      if (!state.board.floor.has(beyond) || state.boxes.has(beyond)) return null;
      const label = state.boxes.get(next);
      result.boxes.delete(next);
      result.boxes.set(beyond, label);
    }
    result.robot = [y + dy, x + dx];
    return result;
  }

  function isPushMove(state, direction) {
    const delta = DIRS[direction];
    if (!delta) return false;
    const [dy, dx] = delta;
    return state.boxes.has(positionKey(state.robot[0] + dy, state.robot[1] + dx));
  }

  function serializeState(state) {
    return {rows: state.board.rows, robot: state.robot, boxes: [...state.boxes]};
  }

  return {
    DIRS,
    positionKey,
    parseRows,
    cloneState,
    isGoal,
    moveState,
    isPushMove,
    serializeState,
  };
});
