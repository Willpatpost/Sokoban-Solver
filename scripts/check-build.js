"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const read = relative => fs.readFileSync(path.join(root, relative), "utf8");
const manifest = JSON.parse(read("docs/build.json"));
assert.match(manifest.build, /^\d{4}-\d{2}-\d{2}\.\d+$/, "invalid docs/build.json revision");

const html = read("docs/index.html");
const bootstrap = read("docs/bootstrap.js");
const app = read("docs/app.js");
const worker = read("docs/solver-worker.js");
const expectedAssets = [
  "levels.js",
  "game-state.js",
  "path-validation.js",
  "keyboard-policy.js",
  "search-log.js",
  "director-policy.js",
  "solver-director.js",
  "app.js",
];

assert.match(html, /<script src="bootstrap\.js"><\/script>/);
assert.doesNotMatch(html, /\?build=/, "index.html must not contain a copied build revision");
assert.doesNotMatch(app, /\d{4}-\d{2}-\d{2}\.\d+/, "app.js must consume the manifest");
assert.match(app, /const SOLVER_BUILD = globalThis\.SOKOMIND_BUILD/);
assert.match(bootstrap, /`styles\.css\$\{revision\}`/);
for (const asset of expectedAssets) {
  assert.match(bootstrap, new RegExp(`["\']${asset.replace(".", "\\.")}["\']`));
}
assert.match(bootstrap, /fetch\("build\.json", \{cache: "no-store"\}\)/);
assert.match(bootstrap, /`\?build=\$\{encodeURIComponent\(manifest\.build\)\}`/);
assert.match(worker, /globalThis\.location\?\.search/);
assert.match(worker, /`solver-engine\.js\$\{engineRevision\}`/);
assert.match(worker, /`solver-search\.js\$\{engineRevision\}`/);

process.stdout.write(`Build manifest ${manifest.build} controls every browser asset revision.\n`);
