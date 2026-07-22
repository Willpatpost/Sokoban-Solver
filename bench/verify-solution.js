"use strict";

const fs = require("node:fs");
const path = require("node:path");
const {LEVELS} = require("../docs/levels.js");
const {replayPath} = require("./evaluator.js");

const MOVE_LINE = /^\s*(\d+)\.\s+(Up|Down|Left|Right)(\s+\(push\))?\s*$/;

function parseSolutionText(text) {
  const moves = [];
  const annotatedPushes = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim() || !/^\s*\d+\./.test(line)) continue;
    const match = MOVE_LINE.exec(line);
    if (!match) throw new Error(`Invalid solution line: ${line}`);
    const expected = moves.length + 1;
    const number = Number(match[1]);
    if (number !== expected) {
      throw new Error(`Expected move ${expected}, found ${number}.`);
    }
    moves.push(match[2]);
    if (match[3]) annotatedPushes.push(number);
  }
  if (!moves.length) throw new Error("Solution file contains no numbered moves.");
  return {moves, annotatedPushes};
}

function isGoalState(state) {
  return [...state.boxes].every(([position, label]) =>
    state.goalsByLabel.get(label)?.includes(position));
}

function verifySolution(rows, solutionText) {
  const parsed = parseSolutionText(solutionText);
  const replay = replayPath(rows, parsed.moves);
  if (!replay.valid) {
    throw new Error(`Illegal move ${replay.index + 1}: ${replay.reason}.`);
  }
  if (!isGoalState(replay.state)) throw new Error("Replay is legal but does not solve the puzzle.");
  if (parsed.annotatedPushes.length && parsed.annotatedPushes.length !== replay.pushes) {
    throw new Error(
      `Solution annotates ${parsed.annotatedPushes.length} pushes, but replay performs ${replay.pushes}.`,
    );
  }
  return {moves: replay.moves, pushes: replay.pushes};
}

function usage() {
  return "Usage: node bench/verify-solution.js LEVEL SOLUTION_FILE";
}

function main(argv = process.argv.slice(2)) {
  if (argv.length !== 2 || argv.includes("--help")) {
    process.stdout.write(`${usage()}\n`);
    return argv.includes("--help") ? 0 : 2;
  }
  const [level, solutionFile] = argv;
  const rows = LEVELS[level];
  if (!rows) throw new Error(`Unknown level: ${level}`);
  const text = fs.readFileSync(path.resolve(solutionFile), "utf8");
  const result = verifySolution(rows, text);
  process.stdout.write(
    `Verified ${level}: ${result.moves} moves, ${result.pushes} pushes.\n`,
  );
  return 0;
}

if (require.main === module) {
  try {
    process.exitCode = main();
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}

module.exports = {isGoalState, parseSolutionText, verifySolution};
