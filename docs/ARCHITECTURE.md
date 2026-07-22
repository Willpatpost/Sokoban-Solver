# Sokomind solver architecture

## General-purpose search rule

Sokomind's search algorithms must derive every decision from the supplied board
and current state. Solver production code must not contain a saved solution,
level-specific coordinate, level-name branch, or heuristic tuned to recognize a
particular built-in puzzle. Built-in levels are examples and regression cases,
not privileged solver inputs.

The saved Huge route is a diagnostic replay. It is useful for identifying legal
states that pruning must preserve, measuring heuristic humps along a difficult
solution, and establishing an incumbent for tests. The solver never loads that
route and must remain correct when it is absent.

## Runtime components

- `Searches/Sokomind.py` contains the Python parser and search implementations.
- `docs/solver-worker.js` contains the browser search kernels and board-derived
  analysis used by Web Workers.
- `docs/app.js` owns gameplay, worker scheduling, replay validation, and UI state.
- `shared/sokomind-conformance.json` is the canonical built-in level catalog and
  cross-runtime rule fixture.
- `bench/` runs isolated, replay-validated search and solution checks.

Hard pruning requires independent differential evidence. A saved solution may
prove that one route is retained, but it cannot establish that a pruning rule is
safe for arbitrary puzzles; generated and exhaustive state families provide that
broader evidence.
