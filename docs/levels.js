(function exposeSokomindLevels(root) {
  const EMBEDDED_LEVELS = {
    "ultra-tiny": ["OOOOO", "O R O", "O A O", "O a O", "OOOOO"],
    tiny: ["OOOOOO", "O R  O", "O XO O", "OO A O", "OSa  O", "OOOOOO"],
    medium: ["OOOOOOO", "Oa   bO", "O AXB O", "O XRX O", "OSCXDSO", "OcS SdO", "OOOOOOO"],
    large: ["OOOOOOOOOO", "OOOOOOOSSO", "OOOOO  abO", "OOOOO XSSO", "OOOOOO  OO",
      "OR     OOO", "OO A X X O", "OO BXO O O", "OO   O   O", "OOOOOOOOOO"],
    huge: ["OOOOOOOOOOOOOOO", "OaSS   S   SSbO", "OSCS  OOO  SDSO", "OX X  OOO  X XO",
      "O     OOO     O", "OOOO   X   OOOO", "O      O      O", "O G hOOOOOH g O",
      "O      O      O", "OOO         OOO", "OOO   X X   OOO", "OOOOOOOROOOOOOO",
      "O B X X X X A O", "O Sc       dS O", "OOOOOOOOOOOOOOO"],
  };
  const LEVELS = typeof module !== "undefined" && module.exports
    ? require("../shared/sokomind-conformance.json").levels
    : EMBEDDED_LEVELS;

  function stateFromRows(rows) {
    let robot = null;
    const boxes = [];
    rows.forEach((row, y) => [...row].forEach((cell, x) => {
      if (cell === "R") robot = [y, x];
      if (cell === "X" || (/[A-Z]/.test(cell) && !"ORS".includes(cell))) {
        boxes.push([`${y},${x}`, cell]);
      }
    }));
    return {rows: [...rows], robot, boxes};
  }

  const api = {LEVELS, EMBEDDED_LEVELS, stateFromRows};
  root.SokomindLevels = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis);
