const assert = require("node:assert/strict");
const test = require("node:test");
const {compareSuite} = require("./performance-gate.js");

const baseline = {
  expected: {
    cases: 2,
    solved: 2,
    totalVisited: 100,
    totalGenerated: 200,
    totalRetained: 80,
    maxFrontier: 20,
    transpositionEvictions: 0,
  },
  tolerance: {
    totalVisited: 0.05,
    totalGenerated: 0.05,
    totalRetained: 0.05,
    maxFrontier: 0.1,
    transpositionEvictions: 0,
  },
  memory: {reviewedPeakBytes: 1000, maximumFactor: 2},
};

function actual(overrides = {}) {
  return {
    cases: 2,
    solved: 2,
    valid: true,
    errors: 0,
    totalVisited: 100,
    totalGenerated: 200,
    totalRetained: 80,
    maxFrontier: 20,
    transpositionEvictions: 0,
    peakHeapBytes: 1000,
    elapsedMs: 999999,
    ...overrides,
  };
}

test("deterministic gate accepts reviewed counters and ignores elapsed time", () => {
  assert.deepEqual(compareSuite(actual(), baseline), []);
});

test("deterministic gate rejects material state and eviction regressions", () => {
  const failures = compareSuite(
    actual({totalVisited: 106, transpositionEvictions: 1}),
    baseline,
  );
  assert.equal(failures.length, 2);
  assert.match(failures.join("\n"), /totalVisited/);
  assert.match(failures.join("\n"), /transpositionEvictions/);
});

test("memory has a separate machine-sensitive tolerance", () => {
  assert.match(
    compareSuite(actual({peakHeapBytes: 2001}), baseline).join("\n"),
    /machine-sensitive/,
  );
});
