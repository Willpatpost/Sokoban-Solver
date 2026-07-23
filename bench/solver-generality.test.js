"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");
const PRODUCTION_SOLVER_FILES = [
  "docs/solver-worker.js",
  "docs/solver-engine.js",
  "docs/solver-search.js",
  "Searches/Sokomind.py",
];
const BUILTIN_LEVEL_NAMES = ["ultra-tiny", "tiny", "medium", "large", "huge"];

test("production solvers do not consume saved routes or branch on built-in levels", () => {
  for (const relativePath of PRODUCTION_SOLVER_FILES) {
    const source = fs.readFileSync(path.join(ROOT, relativePath), "utf8");
    assert.doesNotMatch(source, /optimalForHuge|HUGE_SOLUTION|known solution/i, relativePath);
    for (const level of BUILTIN_LEVEL_NAMES) {
      const levelBranch = new RegExp(
        `(?:===?|!==?)\\s*["']${level}["']|["']${level}["']\\s*(?:===?|!==?)`,
        "i",
      );
      assert.doesNotMatch(source, levelBranch, `${relativePath} must not special-case ${level}`);
    }
  }
});
