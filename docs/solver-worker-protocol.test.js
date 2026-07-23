const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function loadProtocol(search) {
  const messages = [];
  const context = {
    location: {search: "?build=test"},
    importScripts: () => {},
    postMessage: message => messages.push(message),
    search,
    bidirectionalSide: () => {},
    Error,
  };
  vm.runInNewContext(
    fs.readFileSync(path.join(__dirname, "solver-worker.js"), "utf8"),
    context,
    {filename: "solver-worker.js"},
  );
  return {context, messages};
}

test("worker protocol preserves structured terminal results", () => {
  const {context, messages} = loadProtocol(() => ({
    path: ["Down"],
    status: "solved",
    terminationReason: "solution",
    visited: 2,
  }));
  context.onmessage({data: {algorithm: "push-astar"}});
  assert.equal(messages.length, 1);
  assert.equal(messages[0].type, "done");
  assert.equal(messages[0].status, "solved");
  assert.equal(messages[0].terminationReason, "solution");
});

test("worker protocol converts thrown failures into an explicit failed result", () => {
  const {context, messages} = loadProtocol(() => {
    throw new Error("synthetic worker crash");
  });
  context.onmessage({data: {algorithm: "push-astar"}});
  assert.equal(messages.length, 1);
  assert.equal(messages[0].status, "failed");
  assert.equal(messages[0].terminationReason, "worker-exception");
  assert.match(messages[0].error, /synthetic worker crash/);
  assert.equal(messages[0].path, null);
});
