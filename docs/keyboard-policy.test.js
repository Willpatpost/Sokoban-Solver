const assert = require("node:assert/strict");
const test = require("node:test");

const {
  IGNORED_GAME_SHORTCUT_TARGETS,
  shouldIgnoreGameShortcut,
} = require("./keyboard-policy.js");

function targetMatching(match) {
  return {
    closest(selector) {
      assert.equal(selector, IGNORED_GAME_SHORTCUT_TARGETS);
      return match ? this : null;
    },
  };
}

test("game shortcuts ignore editable and interactive targets", () => {
  assert.equal(shouldIgnoreGameShortcut(targetMatching(true)), true);
});

test("game shortcuts remain active for the board and non-elements", () => {
  assert.equal(shouldIgnoreGameShortcut(targetMatching(false)), false);
  assert.equal(shouldIgnoreGameShortcut(null), false);
  assert.equal(shouldIgnoreGameShortcut({}), false);
});
