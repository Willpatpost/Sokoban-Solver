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
  assert.match(html, /id="copy-search-json"/);
  assert.match(html, /levels\.js\?build=[\s\S]*app\.js\?build=/);
  assert.match(html, /director-policy\.js/);
  assert.match(html, /keyboard-policy\.js[\s\S]*app\.js/);
  assert.match(html, /id="solver-build"/);
  const build = app.match(/const SOLVER_BUILD = "([^"]+)";/)?.[1];
  assert.ok(build);
  assert.match(html, new RegExp(`director-policy\\.js\\?build=${build.replaceAll(".", "\\.")}`));
  assert.match(app, /\$\("solver-build"\)\.textContent = SOLVER_BUILD/);
  assert.match(html, new RegExp(`app\\.js\\?build=${build.replaceAll(".", "\\.")}`));
  assert.match(app, /function appendSearchLog\(/);
  assert.match(app, /algorithm: "analyze-puzzle"/);
  assert.match(app, /copy-search-log/);
  assert.match(app, /searchLogJsonLines/);
  assert.match(app, /SokomindKeyboard\.shouldIgnoreGameShortcut\(event\.target\)/);
});

test("Ultimate scheduling retires stale phases and reclaims silent workers", () => {
  const app = fs.readFileSync(path.join(__dirname, "app.js"), "utf8");

  assert.match(app, /const directQueue = \[\.\.\.evacuationPlans, \.\.\.beamPlans/);
  assert.match(app, /retirePendingPlans\(/);
  assert.match(app, /packing checkpoint superseded opening and bridge exploration/);
  assert.match(app, /silentSeconds >= 120/);
  assert.match(app, /abandonWorker\("watchdog"\)/);
  assert.match(app, /Recovering silent discovery worker/);
  assert.match(app, /watchdogRecovery/);
  assert.match(app, /sequenceMacros: false/);
  assert.match(app, /bridgeOutstanding = Math\.max\(0, bridgeOutstanding - 1\)/);
  assert.match(app, /bridgeCampaignViable/);
  assert.match(app, /Candidate landmark bridges queued/);
  assert.match(app, /Promising bridge checkpoint promoted/);
  assert.match(app, /activeBridgeWorkers > 0 \|\| lastQueuedDirectKind === "bridge"/);
  assert.match(app, /requiredAlternative/);
  assert.match(app, /maxWorkerConcurrency - activeSideWorkers/);
  assert.match(app, /persistent partitioned exact contour/);
  assert.match(app, /exactShard: \{index, count: exactRoundShardCount, depth: 4\}/);
  assert.match(app, /requiredWork\.isComplete\(\)/);
  assert.match(app, /Bridge campaign circuit breaker opened/);
  assert.match(app, /Started anytime checkpoint discovery/);
  assert.match(app, /anytimeGuided/);
  assert.match(app, /Refilled exact-phase discovery capacity/);
  assert.match(app, /const anytimeAttempts = new Map\(\)/);
  assert.match(app, /\(anytimeAttempts\.get\(candidate\.id\) \|\| 0\) < 2/);
  assert.match(app, /directCapacity = Math\.max\(0, maxWorkerConcurrency - activeSideWorkers\)/);
  assert.doesNotMatch(app, /if \(settled \|\| activeEvacuationWorkers > 0\) return/);
  assert.match(app, /exactRoundShardCount = anytimeWorkers[\s\S]*?\? 1/);
  assert.match(app, /provedUnsolvable = exactRoundComplete && !Number\.isFinite/);
  assert.match(app, /discardedExactIncumbent \? Infinity/);
  assert.doesNotMatch(app, /searchLog\.splice\(0/);
  assert.match(app, /searchLog\.slice\(-1500\)/);
});
