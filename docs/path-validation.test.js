const assert = require("node:assert/strict");
const test = require("node:test");

const {validatePathToGoal} = require("./path-validation.js");

const cloneState = state => ({position: state.position});
const moveState = (state, move) => {
  if (move !== "Right" || state.position >= 2) return null;
  return {position: state.position + 1};
};
const isGoal = state => state.position === 2;

test("path validation accepts and trims a path at the goal", () => {
  assert.deepEqual(
    validatePathToGoal({position: 0}, ["Right", "Right", "Right"], cloneState, moveState, isGoal),
    ["Right", "Right"],
  );
});

test("path validation rejects illegal and incomplete paths", () => {
  assert.equal(
    validatePathToGoal({position: 0}, ["Left"], cloneState, moveState, isGoal),
    null,
  );
  assert.equal(
    validatePathToGoal({position: 0}, ["Right"], cloneState, moveState, isGoal),
    null,
  );
});
