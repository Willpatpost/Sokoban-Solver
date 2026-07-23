(function attachSearchLog(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.SokomindSearchLog = api;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  "use strict";

  function text(entries) {
    return entries.map(entry => entry.text).join("\n");
  }

  function jsonLines(entries) {
    return entries.map(entry => JSON.stringify(entry.event)).join("\n");
  }

  function structuredStats(stats) {
    return Object.fromEntries(Object.entries(stats || {}).map(([name, value]) => {
      if (typeof value !== "string") return [name, value];
      const compact = value.replaceAll(",", "");
      if (/^-?\d+(?:\.\d+)?$/.test(compact)) return [name, Number(compact)];
      if (/^-?\d+(?:\.\d+)?\/s$/.test(compact)) {
        return [name, Number(compact.slice(0, -2))];
      }
      if (/^-?\d+(?:\.\d+)?s$/.test(compact)) {
        return [name, Number(compact.slice(0, -1))];
      }
      return [name, value];
    }));
  }

  function formatTime(seconds) {
    const whole = Math.floor(seconds);
    const minutes = Math.floor(whole / 60);
    return `${String(minutes).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")}`;
  }

  function shortStateId(value) {
    let hash = 2166136261;
    for (let index = 0; index < value.length; index++) {
      hash ^= value.charCodeAt(index);
      hash = Math.imul(hash, 16777619) >>> 0;
    }
    return hash.toString(16).padStart(8, "0");
  }

  return {text, jsonLines, structuredStats, formatTime, shortStateId};
});
