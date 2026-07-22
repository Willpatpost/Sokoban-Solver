const assert = require("node:assert/strict");
const test = require("node:test");

const {evaluateCheckpoints, replayPath} = require("./evaluator.js");

const ROWS = ["OOOOOOOO", "O R X SO", "OOOOOOOO"];

test("checkpoint evaluation replays paths and computes its own fixed lower bound", () => {
  const evaluation = evaluateCheckpoints(ROWS, {
    bestEstimate: 0,
    checkpoint: {
      path: ["Right", "Right"],
      state: {robot: [1, 4], boxes: [["1,5", "X"]]},
      estimate: 0,
    },
  });

  assert.equal(evaluation.candidates, 1);
  assert.equal(evaluation.replayValid, 1);
  assert.deepEqual(evaluation.best, {moves: 2, pushes: 1, remainingPushes: 1, projectedPushes: 2});
});

test("checkpoint evaluation rejects illegal or mismatched solver checkpoints", () => {
  const evaluation = evaluateCheckpoints(ROWS, {
    checkpoint: {path: ["Up"], state: {robot: [1, 1], boxes: [["1,4", "X"]]}},
    checkpoints: [
      {path: ["Right"], state: {robot: [1, 3], boxes: [["1,5", "X"]]}},
    ],
  });

  assert.equal(evaluation.replayValid, 0);
  assert.equal(evaluation.rejected, 2);
  assert.equal(evaluation.best, null);
});

test("path replay rejects unknown and mechanically illegal moves", () => {
  assert.equal(replayPath(ROWS, ["Sideways"]).valid, false);
  assert.equal(replayPath(ROWS, ["Up"]).valid, false);
});
