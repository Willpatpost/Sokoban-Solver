(function exposeKeyboardPolicy(root) {
  "use strict";

  const IGNORED_GAME_SHORTCUT_TARGETS =
    'input, textarea, select, button, [contenteditable]:not([contenteditable="false"])';

  function shouldIgnoreGameShortcut(target) {
    return Boolean(
      target &&
      typeof target.closest === "function" &&
      target.closest(IGNORED_GAME_SHORTCUT_TARGETS),
    );
  }

  const api = {IGNORED_GAME_SHORTCUT_TARGETS, shouldIgnoreGameShortcut};
  root.SokomindKeyboard = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis);
