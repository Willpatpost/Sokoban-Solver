const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const {validatePathToGoal} = require("./path-validation.js");

const cloneState = state => ({position: state.position});
const moveState = (state, move) => {
  if (move !== "Right" || state.position >= 2) return null;
  return {position: state.position + 1};
};
const isGoal = state => state.position === 2;

test("path validation accepts and trims a path at the goal", () => {
  assert.deepEqual(
    validatePathToGoal({position: 0}, ["Right", "Right", "Right"], cloneState, moveState, isGoal),
    ["Right", "Right"],
  );
});

test("path validation rejects illegal and incomplete paths", () => {
  assert.equal(
    validatePathToGoal({position: 0}, ["Left"], cloneState, moveState, isGoal),
    null,
  );
  assert.equal(
    validatePathToGoal({position: 0}, ["Right"], cloneState, moveState, isGoal),
    null,
  );
});

test("web UI exposes a separate copyable search log", () => {
  const html = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
  const app = fs.readFileSync(path.join(__dirname, "app.js"), "utf8");

  assert.match(html, /id="search-log-count"/);
  assert.match(html, /id="search-log-text"/);
  assert.match(html, /id="copy-search-log"/);
  assert.match(app, /function appendSearchLog\(/);
  assert.match(app, /algorithm: "analyze-puzzle"/);
  assert.match(app, /copy-search-log/);
});

test("Ultimate scheduling retires stale phases and reclaims silent workers", () => {
  const app = fs.readFileSync(path.join(__dirname, "app.js"), "utf8");

  assert.match(app, /const directPlans = \[\.\.\.evacuationPlans, \.\.\.beamPlans/);
  assert.match(app, /retirePendingPlans\(/);
  assert.match(app, /packing checkpoint superseded opening exploration/);
  assert.match(app, /silentSeconds >= 120/);
  assert.match(app, /abandonWorker\("watchdog"\)/);
  assert.match(app, /bridgeOutstanding = Math\.max\(0, bridgeOutstanding - 1\)/);
  assert.match(app, /provedUnsolvable = exhaustedExactBound && !Number\.isFinite/);
  assert.match(app, /discardedExactIncumbent \? Infinity/);
});
