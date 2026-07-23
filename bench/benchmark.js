const {spawn} = require("node:child_process");
const path = require("node:path");
const {performance} = require("node:perf_hooks");
const {LEVELS} = require("../docs/levels.js");
const {GENERATED_CASES, mirrorRows, rotateRows} = require("./generated-cases.js");

const CASE_RUNNER = path.join(__dirname, "case-runner.js");
const HUGE_STRUCTURAL_PAYLOAD = {
  maxDepth: 460,
  maxVisited: 6000,
  transpositionLimit: 60000,
  planBeamWidth: 40,
  planBoxBranches: 6,
  maxPlanSegments: 160,
  planSlack: 240,
  sequenceMacroLimit: 24,
  sequenceMacroExplored: 48,
  sequenceMacroResults: 4,
  targetedMacroExplored: 96,
  progressIntervalMs: 5000,
};

const SUITES = {
  smoke: [
    {
      name: "ultra-tiny push astar",
      level: "ultra-tiny",
      algorithm: "push-astar",
      timeoutMs: 5000,
      weight: 1,
    },
    {
      name: "tiny portfolio",
      level: "tiny",
      algorithm: "portfolio",
      timeoutMs: 10000,
      weight: 2,
      payload: {maxVisited: 50000},
    },
    {
      name: "medium guided beam",
      level: "medium",
      algorithm: "push-beam",
      timeoutMs: 15000,
      weight: 4,
      payload: {maxVisited: 120000, beamWidth: 240, maxDepth: 120, progressInterval: 10000},
    },
  ],
  alpha: [
    {
      name: "ultra-tiny push astar",
      level: "ultra-tiny",
      algorithm: "push-astar",
      timeoutMs: 5000,
      weight: 1,
    },
    {
      name: "tiny portfolio",
      level: "tiny",
      algorithm: "portfolio",
      timeoutMs: 10000,
      weight: 2,
      payload: {maxVisited: 60000},
    },
    {
      name: "medium guided beam",
      level: "medium",
      algorithm: "push-beam",
      timeoutMs: 20000,
      weight: 4,
      payload: {
        maxVisited: 180000,
        beamWidth: 360,
        maxDepth: 180,
        progressInterval: 15000,
        sequenceMacros: true,
      },
    },
    {
      name: "large guided beam",
      level: "large",
      algorithm: "push-beam",
      timeoutMs: 45000,
      weight: 8,
      payload: {
        maxVisited: 350000,
        beamWidth: 500,
        maxDepth: 220,
        progressInterval: 25000,
        sequenceMacros: true,
        continuationVisited: 70000,
        endgameVisited: 70000,
      },
    },
    {
      name: "huge checkpoint smoke",
      level: "huge",
      algorithm: "push-beam",
      timeoutMs: 30000,
      weight: 16,
      payload: {
        maxVisited: 2000,
        beamWidth: 300,
        maxDepth: 260,
        progressInterval: 1000,
        sequenceMacros: false,
        checkpointLimit: 16,
      },
    },
    ...GENERATED_CASES,
  ],
  validation: GENERATED_CASES,
  huge: [
    {
      name: "huge structural plan",
      level: "huge",
      algorithm: "plan-macro-beam",
      timeoutMs: 300000,
      weight: 64,
      payload: HUGE_STRUCTURAL_PAYLOAD,
    },
    {
      name: "huge mirrored structural plan",
      rows: mirrorRows(LEVELS.huge),
      algorithm: "plan-macro-beam",
      timeoutMs: 300000,
      weight: 64,
      payload: HUGE_STRUCTURAL_PAYLOAD,
    },
    {
      name: "huge rotated structural plan",
      rows: rotateRows(LEVELS.huge),
      algorithm: "plan-macro-beam",
      timeoutMs: 300000,
      weight: 64,
      payload: HUGE_STRUCTURAL_PAYLOAD,
    },
  ],
};

function parseArgs(argv) {
  const options = {suite: "smoke", jsonl: false};
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === "--suite") options.suite = argv[++index];
    else if (arg === "--level") options.level = argv[++index];
    else if (arg === "--algorithm") options.algorithm = argv[++index];
    else if (arg === "--max-visited") options.maxVisited = Number(argv[++index]);
    else if (arg === "--timeout-ms") options.timeoutMs = Number(argv[++index]);
    else if (arg === "--jsonl") options.jsonl = true;
    else if (arg === "--help") options.help = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return options;
}

function usage() {
  return [
    "Usage: node bench/benchmark.js [options]",
    "",
    "Options:",
    "  --suite smoke|alpha|validation|huge  Benchmark suite to run (default: smoke).",
    "  --level LEVEL            Run one level from docs/levels.js.",
    "  --algorithm ALGORITHM    Algorithm for --level (default: push-beam).",
    "  --max-visited N          Worker state budget for a single-level run.",
    "  --timeout-ms N           Child-process timeout for a single-level run.",
    "  --jsonl                  Emit each case result before the final summary.",
  ].join("\n");
}

function caseScore(result, weight) {
  if (result.timeout || result.error || !result.valid) return -1000000 * weight;
  const visitedCost = (result.visited || 0) * 0.02;
  const timeCost = (result.elapsedMs || 0) * 2;
  const pathCost = (result.moves || 0) + 5 * (result.pushes || 0);
  const checkpoint = result.checkpointEvaluation?.best;
  const partialCredit = checkpoint
    ? Math.max(0, 100000 - 600 * checkpoint.remainingPushes - 4 * checkpoint.pushes)
    : 0;
  return Math.round(weight * (
    result.solved ? 1000000 - visitedCost - timeCost - pathCost : partialCredit - visitedCost - timeCost
  ));
}

function runChild(caseSpec) {
  return new Promise(resolve => {
    const timeout = caseSpec.timeoutMs || 30000;
    const started = performance.now();
    const child = spawn(process.execPath, [
      CASE_RUNNER,
      JSON.stringify({...caseSpec, streamProgress: true}),
    ], {stdio: ["ignore", "pipe", "pipe"]});
    let stdout = "", stderr = "", settled = false, lastProgress = null, finalResult = null;
    let firstOutputAt = null, resultAt = null;
    const timer = setTimeout(() => {
      if (!settled) child.kill();
    }, timeout);
    child.stdout.on("data", chunk => {
      if (firstOutputAt === null) firstOutputAt = performance.now();
      stdout += chunk;
      const lines = stdout.split(/\r?\n/);
      stdout = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        let event;
        try {
          event = JSON.parse(line);
        } catch (_error) {
          continue;
        }
        if (event.type === "progress") lastProgress = event.message;
        else if (event.type === "result") {
          finalResult = event.result;
          resultAt = performance.now();
        }
        else if (event.type === "error") {
          finalResult = {
            name: caseSpec.name,
            level: caseSpec.level,
            algorithm: caseSpec.algorithm,
            error: event.error,
            stack: event.stack,
            valid: false,
            solved: false,
          };
          resultAt = performance.now();
        }
      }
    });
    child.stderr.on("data", chunk => {
      stderr += chunk;
    });
    child.on("close", (code, signal) => {
      const closedAt = performance.now();
      clearTimeout(timer);
      settled = true;
      if (signal) {
        resolve({
          name: caseSpec.name,
          level: caseSpec.level,
          algorithm: caseSpec.algorithm,
          timeout: true,
          timeoutMs: timeout,
          elapsedMs: Math.round(closedAt - started),
          valid: true,
          solved: false,
          visited: lastProgress?.visited || 0,
          bestEstimate: lastProgress?.bestEstimate,
          bestPushes: lastProgress?.bestPushes,
          progress: {messages: lastProgress ? 1 : 0, last: lastProgress},
          processLifecycle: {
            spawnToFirstOutputMs: firstOutputAt === null ? null :
              Math.round((firstOutputAt - started) * 1000) / 1000,
            spawnToResultMs: null,
            shutdownMs: null,
            totalMs: Math.round((closedAt - started) * 1000) / 1000,
          },
          stderr: stderr.trim(),
        });
        return;
      }
      if (!finalResult && stdout.trim()) {
        try {
          const event = JSON.parse(stdout.trim());
          finalResult = event.type === "result" ? event.result : event;
          resultAt = closedAt;
        } catch (parseError) {
          finalResult = {
            name: caseSpec.name,
            level: caseSpec.level,
            algorithm: caseSpec.algorithm,
            error: `Invalid case-runner JSON: ${parseError.message}`,
            stdout,
            stderr,
            valid: false,
            solved: false,
          };
        }
      }
      const result = finalResult || {
          name: caseSpec.name,
          level: caseSpec.level,
          algorithm: caseSpec.algorithm,
          error: `Case runner exited without a result (code ${code})`,
          stdout,
          stderr,
          valid: false,
          solved: false,
      };
      if (code && !result.error) result.error = `Case runner exited with code ${code}`;
      if (stderr.trim()) result.stderr = stderr.trim();
      result.processLifecycle = {
        spawnToFirstOutputMs: firstOutputAt === null ? null :
          Math.round((firstOutputAt - started) * 1000) / 1000,
        spawnToResultMs: resultAt === null ? null :
          Math.round((resultAt - started) * 1000) / 1000,
        shutdownMs: resultAt === null ? null :
          Math.round((closedAt - resultAt) * 1000) / 1000,
        totalMs: Math.round((closedAt - started) * 1000) / 1000,
      };
      resolve(result);
    });
  });
}

function buildCases(options) {
  if (options.level) {
    return [{
      name: `${options.level} ${options.algorithm || "push-beam"}`,
      level: options.level,
      algorithm: options.algorithm || "push-beam",
      timeoutMs: options.timeoutMs || 30000,
      weight: 1,
      payload: {
        ...(options.maxVisited ? {maxVisited: options.maxVisited} : {}),
        progressInterval: 25000,
      },
    }];
  }
  const suite = SUITES[options.suite];
  if (!suite) throw new Error(`Unknown suite: ${options.suite}`);
  return suite;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    process.stdout.write(`${usage()}\n`);
    return;
  }
  const cases = buildCases(options);
  const started = performance.now();
  const results = [];
  for (const caseSpec of cases) {
    const result = await runChild(caseSpec);
    result.weight = caseSpec.weight || 1;
    result.score = caseScore(result, result.weight);
    results.push(result);
    if (options.jsonl) process.stdout.write(`${JSON.stringify({type: "case", ...result})}\n`);
  }
  const summary = {
    schemaVersion: 2,
    suite: options.level ? "single" : options.suite,
    elapsedMs: Math.round(performance.now() - started),
    attempted: results.length,
    solved: results.filter(result => result.solved).length,
    valid: results.every(result => result.valid),
    timeouts: results.filter(result => result.timeout).length,
    errors: results.filter(result => result.error).length,
    totalVisited: results.reduce((sum, result) => sum + (result.visited || 0), 0),
    memorySupportedCases: results.filter(result => result.performance?.heapSupported).length,
    peakHeapBytes: results.reduce((peak, result) =>
      Math.max(peak, result.performance?.heapPeakBytes || 0), 0),
    memory: {
      supportedCases: results.filter(result => result.performance?.memory?.supported).length,
      peakBytes: results.reduce((peak, result) =>
        Math.max(peak, result.performance?.memory?.peakBytes || 0), 0),
      gcControlled: false,
    },
    totalChildProcessMs: Math.round(results.reduce((sum, result) =>
      sum + (result.processLifecycle?.totalMs || 0), 0) * 1000) / 1000,
    totalShutdownMs: Math.round(results.reduce((sum, result) =>
      sum + (result.processLifecycle?.shutdownMs || 0), 0) * 1000) / 1000,
    totalScore: results.reduce((sum, result) => sum + result.score, 0),
    cases: results,
  };
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
  if (!summary.valid || summary.errors) process.exitCode = 1;
}

if (require.main === module) {
  main().catch(error => {
    process.stderr.write(`${error.stack || error.message}\n`);
    process.exitCode = 1;
  });
}

module.exports = {buildCases, caseScore, parseArgs, runChild, SUITES};
