const assert = require("node:assert/strict");
const test = require("node:test");

const {caseScore, runChild} = require("./benchmark.js");

test("unsolved benchmark scoring ignores solver-reported estimates", () => {
  const base = {
    valid: true,
    solved: false,
    visited: 100,
    elapsedMs: 10,
    checkpointEvaluation: {
      best: {pushes: 3, remainingPushes: 7, projectedPushes: 10},
    },
  };
  assert.equal(
    caseScore({...base, bestEstimate: 0, bestPushes: 0}, 2),
    caseScore({...base, bestEstimate: 9999, bestPushes: 9999}, 2),
  );
});

test("unsolved searches receive no partial credit without a validated checkpoint", () => {
  const score = caseScore({
    valid: true,
    solved: false,
    visited: 100,
    elapsedMs: 10,
    bestEstimate: 0,
    checkpointEvaluation: {best: null},
  }, 1);
  assert.equal(score, -22);
});

test("isolated benchmark cases report heap and process lifecycle telemetry", async () => {
  const result = await runChild({
    name: "telemetry fixture",
    rows: ["OOOOO", "O R O", "O X O", "O S O", "OOOOO"],
    algorithm: "push-astar",
    timeoutMs: 5000,
  });
  assert.equal(result.valid, true);
  assert.equal(result.solved, true);
  assert.equal(result.status, "solved");
  assert.equal(result.terminationReason, "solution");
  assert.equal(result.performance.heapSupported, true);
  assert.ok(result.performance.heapUsedBytes > 0);
  assert.ok(result.performance.heapPeakBytes >= result.performance.heapUsedBytes);
  assert.ok(result.performance.heapSamples >= 2);
  assert.ok(result.runnerLifecycle.workerLoadMs >= 0);
  assert.ok(result.runnerLifecycle.searchMs >= 0);
  assert.ok(result.runnerLifecycle.totalMs >= result.runnerLifecycle.searchMs);
  assert.equal(typeof result.runnerLifecycle.explicitGcAvailable, "boolean");
  assert.ok(result.processLifecycle.spawnToFirstOutputMs >= 0);
  assert.ok(result.processLifecycle.spawnToResultMs >= 0);
  assert.ok(result.processLifecycle.shutdownMs >= 0);
  assert.ok(result.processLifecycle.totalMs >= result.processLifecycle.spawnToResultMs);
});
