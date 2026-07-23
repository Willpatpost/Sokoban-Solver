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
- `docs/app.js` owns browser UI state, rendering, controls, timing, and animation.
- `docs/game-state.js` is the independently testable browser gameplay/rules core.
- `docs/search-log.js` provides pure readable and structured telemetry formatting.
- `docs/solver-director.js` owns worker portfolios, checkpoint handoffs, lifecycle
  accounting, replay validation, and exact-search transitions.
- `docs/solver-worker.js` is the stable Web Worker protocol entry point. It carries
  the page's build query into each implementation module.
- `docs/solver-engine.js` owns browser board parsing, state identities, topology,
  heuristics, deadlocks, dependencies, local analysis, and push generation.
- `docs/solver-search.js` owns browser search algorithms, reconstruction,
  checkpoints, progress messages, and result telemetry.
- `shared/sokomind-conformance.json` is the canonical built-in level catalog and
  cross-runtime rule fixture.
- `bench/` runs isolated, replay-validated search and solution checks.

Browser files remain classic scripts so the dependency-free GitHub Pages build and
Web Workers need no bundler. The HTML load order supplies pure modules and policies
before the director and UI. Worker imports load the engine before search algorithms.
Node tests evaluate the engine and search module together in the same order.

Hard pruning requires independent differential evidence. A saved solution may
prove that one route is retained, but it cannot establish that a pruning rule is
safe for arbitrary puzzles; generated and exhaustive state families provide that
broader evidence.
