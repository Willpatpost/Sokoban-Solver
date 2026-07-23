const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function loadDirectorStorage({maximumBytes = Infinity, initial = {}} = {}) {
  const values = new Map();
  for (const [key, value] of Object.entries(initial)) values.set(key, String(value));
  const localStorage = {
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => {
      const encoded = String(value);
      if (encoded.length > maximumBytes) throw new Error("QuotaExceededError");
      values.set(key, encoded);
    },
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
    {...checkpoint, storageSavedAt: context.loadExactCheckpoint(
      state, checkpoint.exactShard, Infinity,
    ).storageSavedAt},
  );
  assert.equal(context.loadExactCheckpoint(state, checkpoint.exactShard, 20), null);
  assert.equal(context.loadExactCheckpoint(
    {...state, robot: [1, 1]}, checkpoint.exactShard, Infinity,
  ), null);
});

test("checkpoint storage removes stale builds and retains a bounded recent set", () => {
  const {context, values} = loadDirectorStorage({
    initial: {
      "sokomind-exact-checkpoints-v1": JSON.stringify({
        "old-problem:0/1": {
          problemHash: "old-problem",
          solverBuild: "old-build",
          exactShard: {index: 0, count: 1},
          storageSavedAt: 1,
        },
      }),
    },
  });
  for (let index = 0; index < 12; index++) {
    assert.equal(context.saveExactCheckpoint({
      problemHash: `problem-${index}`,
      solverBuild: "test-build",
      exactShard: {index: 0, count: 1},
      upperBound: "Infinity",
    }), true);
  }
  const stored = JSON.parse(values.get("sokomind-exact-checkpoints-v1"));
  assert.equal(Object.keys(stored).length, 8);
  assert.equal(stored["old-problem:0/1"], undefined);
  assert.ok(stored["problem-11:0/1"]);
});

test("checkpoint storage evicts older entries on quota pressure and reports failure", () => {
  const {context, values} = loadDirectorStorage({maximumBytes: 360});
  const checkpoint = index => ({
    problemHash: `problem-${index}`,
    solverBuild: "test-build",
    exactShard: {index: 0, count: 1},
    upperBound: "Infinity",
    stack: [{payload: "x".repeat(80)}],
  });
  assert.equal(context.saveExactCheckpoint(checkpoint(1)), true);
  assert.equal(context.saveExactCheckpoint(checkpoint(2)), true);
  const stored = JSON.parse(values.get("sokomind-exact-checkpoints-v1"));
  assert.equal(Object.keys(stored).length, 1);
  assert.ok(stored["problem-2:0/1"]);
  assert.equal(context.saveExactCheckpoint({...checkpoint(3), stack: [{payload: "x".repeat(500)}]}), false);
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
