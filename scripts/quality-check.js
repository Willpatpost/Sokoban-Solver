"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {spawnSync} = require("node:child_process");

const root = path.resolve(__dirname, "..");
const roots = ["bench", "docs", "scripts", path.join("tests", "browser")];

function filesUnder(relative) {
  const absolute = path.join(root, relative);
  if (!fs.existsSync(absolute)) return [];
  return fs.readdirSync(absolute, {withFileTypes: true}).flatMap(entry => {
    const child = path.join(relative, entry.name);
    return entry.isDirectory() ? filesUnder(child) : [child];
  });
}

const files = roots.flatMap(filesUnder).sort();
const javascript = files.filter(file => file.endsWith(".js"));
const data = files.filter(file => file.endsWith(".json"));
const formattingErrors = [];

for (const file of files) {
  const source = fs.readFileSync(path.join(root, file), "utf8");
  source.split(/\r?\n/).forEach((line, index) => {
    if (/[ \t]+$/.test(line)) formattingErrors.push(`${file}:${index + 1}: trailing space`);
    if (/^\t/.test(line)) formattingErrors.push(`${file}:${index + 1}: leading tab`);
  });
}
assert.deepEqual(formattingErrors, [], formattingErrors.join("\n"));

for (const file of data) JSON.parse(fs.readFileSync(path.join(root, file), "utf8"));
for (const file of javascript) {
  const checked = spawnSync(process.execPath, ["--check", file], {
    cwd: root,
    encoding: "utf8",
  });
  assert.equal(checked.status, 0, checked.stderr || `${file} failed syntax validation`);
}

process.stdout.write(
  `Quality check passed for ${javascript.length} JavaScript and ${data.length} JSON files.\n`,
);
