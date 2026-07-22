const assert = require("node:assert/strict");
const test = require("node:test");

const {caseScore} = require("./benchmark.js");

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
