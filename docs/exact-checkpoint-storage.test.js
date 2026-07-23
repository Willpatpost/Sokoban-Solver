const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function loadDirectorStorage() {
  const values = new Map();
  const localStorage = {
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: key => values.delete(key),
  };
  const context = {localStorage, SOLVER_BUILD: "test-build", console};
  vm.runInNewContext(
    fs.readFileSync(path.join(__dirname, "solver-director.js"), "utf8"),
    context,
    {filename: "solver-director.js"},
  );
  return {context, values};
}

const state = {
  rows: ["OOOOO", "O R O", "O X O", "O S O", "OOOOO"],
  robot: [1, 2],
  boxes: [["2,2", "X"]],
};

test("exact proof checkpoints round-trip only for the matching problem contract", () => {
  const {context} = loadDirectorStorage();
  const problemHash = context.exactCheckpointProblemHash(state);
  const checkpoint = {
    version: 1,
    problemHash,
    solverBuild: "test-build",
    exactShard: {index: 0, count: 2, depth: 4},
    upperBound: "Infinity",
    threshold: 3,
    stack: [],
  };
  assert.equal(context.saveExactCheckpoint(checkpoint), true);
  assert.deepEqual(
    JSON.parse(JSON.stringify(context.loadExactCheckpoint(
      state, checkpoint.exactShard, Infinity,
    ))),
    checkpoint,
  );
  assert.equal(context.loadExactCheckpoint(state, checkpoint.exactShard, 20), null);
  assert.equal(context.loadExactCheckpoint(
    {...state, robot: [1, 1]}, checkpoint.exactShard, Infinity,
  ), null);
});

test("saved exact proof state can be cleared explicitly per puzzle", () => {
  const {context, values} = loadDirectorStorage();
  const problemHash = context.exactCheckpointProblemHash(state);
  context.saveExactCheckpoint({
    version: 1,
    problemHash,
    solverBuild: "test-build",
    exactShard: {index: 0, count: 1, depth: 4},
    upperBound: "Infinity",
  });
  assert.ok(values.size > 0);
  assert.equal(context.clearExactCheckpoints(problemHash), true);
  assert.equal(
    context.loadExactCheckpoint(state, {index: 0, count: 1, depth: 4}, Infinity),
    null,
  );
});
