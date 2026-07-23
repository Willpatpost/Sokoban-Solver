// Stable Web Worker entry point. Search implementation lives in solver-engine.js.
const engineRevision = globalThis.location?.search || "";
importScripts(
  `solver-engine.js${engineRevision}`,
  `solver-search.js${engineRevision}`,
);

onmessage = ({data}) => {
  try {
    if (data.mode === "bidir-forward" || data.mode === "bidir-reverse") {
      bidirectionalSide(data);
    } else {
      postMessage({type: "done", ...search(data)});
    }
  } catch (error) {
    postMessage({
      type: "done",
      path: null,
      status: "failed",
      terminationReason: "worker-exception",
      error: error instanceof Error ? error.message : String(error),
      visited: 0,
    });
  }
};
