// Stable Web Worker entry point. Search implementation lives in solver-engine.js.
const engineRevision = globalThis.location?.search || "";
importScripts(
  `solver-engine.js${engineRevision}`,
  `solver-search.js${engineRevision}`,
);

onmessage = ({data}) => {
  if (data.mode === "bidir-forward" || data.mode === "bidir-reverse") {
    bidirectionalSide(data);
  } else {
    postMessage({type: "done", ...search(data)});
  }
};
