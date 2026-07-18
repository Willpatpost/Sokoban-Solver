(function exposePathValidation(root) {
  function validatePathToGoal(initialState, path, cloneState, moveState, isGoal) {
    let replay = cloneState(initialState);
    const validated = [];
    for (const move of path) {
      const next = moveState(replay, move);
      if (!next) return null;
      replay = next;
      validated.push(move);
      if (isGoal(replay)) return validated;
    }
    return isGoal(replay) ? validated : null;
  }

  const api = {validatePathToGoal};
  root.SokomindPath = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis);
