const baselines = require("./performance-baselines.json");
const {buildCases, runChild} = require("./benchmark.js");

const COUNTERS = [
  "totalVisited",
  "totalGenerated",
  "totalRetained",
  "maxFrontier",
  "transpositionEvictions",
];

function aggregate(results) {
  return {
    cases: results.length,
    solved: results.filter(result => result.solved).length,
    valid: results.every(result => result.valid),
    errors: results.filter(result => result.error).length,
    totalVisited: results.reduce((sum, result) => sum + result.visited, 0),
    totalGenerated: results.reduce((sum, result) => sum + result.generated, 0),
    totalRetained: results.reduce((sum, result) => sum + result.retained, 0),
    maxFrontier: Math.max(0, ...results.map(result => result.peakFrontier)),
    transpositionEvictions: results.reduce(
      (sum, result) => sum + result.transpositionEvictions,
      0,
    ),
    peakHeapBytes: Math.max(
      0,
      ...results.map(result => result.runnerLifecycle?.memory?.peakBytes || 0),
    ),
  };
}

function compareSuite(actual, baseline) {
  const failures = [];
  for (const field of ["cases", "solved"]) {
    if (actual[field] !== baseline.expected[field]) {
      failures.push(
        `${field}: expected ${baseline.expected[field]}, received ${actual[field]}`,
      );
    }
  }
  if (!actual.valid) failures.push("one or more returned paths failed replay validation");
  if (actual.errors) failures.push(`${actual.errors} isolated runner error(s)`);
  for (const field of COUNTERS) {
    const expected = baseline.expected[field];
    const maximum = Math.ceil(expected * (1 + baseline.tolerance[field]));
    if (actual[field] > maximum) {
      failures.push(`${field}: maximum ${maximum}, received ${actual[field]}`);
    }
  }
  const maximumHeap = Math.ceil(
    baseline.memory.reviewedPeakBytes * baseline.memory.maximumFactor,
  );
  if (!actual.peakHeapBytes) {
    failures.push("heap telemetry is unsupported");
  } else if (actual.peakHeapBytes > maximumHeap) {
    failures.push(
      `peakHeapBytes: machine-sensitive maximum ${maximumHeap}, ` +
        `received ${actual.peakHeapBytes}`,
    );
  }
  return failures;
}

async function runSuite(name) {
  const results = [];
  for (const caseSpec of buildCases({suite: name})) {
    results.push(await runChild(caseSpec));
  }
  const actual = aggregate(results);
  return {suite: name, actual, failures: compareSuite(actual, baselines.suites[name])};
}

async function main() {
  const requested = process.argv.slice(2);
  const suites = requested.length ? requested : Object.keys(baselines.suites);
  const reports = [];
  for (const suite of suites) {
    if (!baselines.suites[suite]) throw new Error(`No reviewed baseline for ${suite}`);
    reports.push(await runSuite(suite));
  }
  process.stdout.write(`${JSON.stringify({
    schemaVersion: baselines.schemaVersion,
    reviewedBuild: baselines.reviewedBuild,
    reports,
  }, null, 2)}\n`);
  if (reports.some(report => report.failures.length)) process.exitCode = 1;
}

if (require.main === module) {
  main().catch(error => {
    process.stderr.write(`${error.stack || error.message}\n`);
    process.exitCode = 1;
  });
}

module.exports = {aggregate, compareSuite, runSuite};
