"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const {evaluateHugeResults, readCaseResults} = require("./huge-performance-gate.js");
const baseline = require("./huge-performance-baselines.json");

function passingResult(index) {
  return {
    name: `huge orientation ${index}`,
    solved: true,
    valid: true,
    pushes: baseline.maximumPerCase.pushes,
    visited: baseline.maximumPerCase.visited,
    generated: baseline.maximumPerCase.generated,
    retained: baseline.maximumPerCase.retained,
    peakFrontier: baseline.maximumPerCase.peakFrontier,
    transpositionEvictions: 0,
  };
}

test("Huge gate accepts replay-valid cases within reviewed deterministic ceilings", () => {
  const results = Array.from({length: baseline.expectedCases}, (_, index) =>
    passingResult(index));
  assert.deepEqual(evaluateHugeResults(results), []);
});

test("Huge gate rejects missing, unsolved, timed-out, and regressed cases", () => {
  const results = [passingResult(0), {
    ...passingResult(1),
    solved: false,
    timeout: true,
    visited: baseline.maximumPerCase.visited + 1,
  }];
  const failures = evaluateHugeResults(results);
  assert.match(failures.join("\n"), /cases:/);
  assert.match(failures.join("\n"), /solved:/);
  assert.match(failures.join("\n"), /timed out/);
  assert.match(failures.join("\n"), /visited maximum/);
});

test("Huge gate reads UTF-8 and PowerShell UTF-16 redirected artifacts", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "sokomind-huge-gate-"));
  try {
    const line = `${JSON.stringify({type: "case", ...passingResult(0)})}\n`;
    for (const [encoding, prefix] of [["utf8", ""], ["utf16le", "\uFEFF"]]) {
      const file = path.join(directory, `${encoding}.jsonl`);
      fs.writeFileSync(file, `${prefix}${line}`, encoding);
      assert.equal(readCaseResults(file).length, 1);
    }
  } finally {
    fs.rmSync(directory, {recursive: true, force: true});
  }
});
