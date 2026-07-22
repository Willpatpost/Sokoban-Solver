"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const {LEVELS} = require("../docs/levels.js");
const {parseSolutionText, verifySolution} = require("./verify-solution.js");

test("saved Huge route is a replay-valid diagnostic solution", () => {
  const solution = fs.readFileSync(path.join(__dirname, "../docs/optimalForHuge.txt"), "utf8");
  const result = verifySolution(LEVELS.huge, solution);
  assert.deepEqual(result, {moves: 640, pushes: 250});
});

test("solution verification is level-agnostic", () => {
  const solution = "Example route:\n\n1. Down (push)\n";
  assert.deepEqual(verifySolution(LEVELS["ultra-tiny"], solution), {moves: 1, pushes: 1});
});

test("solution parser rejects gaps and verifier rejects incomplete paths", () => {
  assert.throws(() => parseSolutionText("1. Down\n3. Up\n"), /Expected move 2/);
  assert.throws(() => verifySolution(LEVELS["ultra-tiny"], "1. Right\n"), /does not solve/);
});
