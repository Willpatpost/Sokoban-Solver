# Sokomind Web

The web edition is a responsive, dependency-free browser game ready for GitHub
Pages. It includes five levels from Ultra Tiny through Huge, keyboard play,
undo/reset, a first-move timer, hints, and animated solving in a background Web
Worker.

The move-history panel records every direction, marks pushes, follows
undo/reset, and can copy the complete numbered sequence for sharing.

The default solver is **Ultimate Bidirectional**, an experimental Sokoban-aware
search. It starts one forward Web Worker from the current board and one or more
reverse Web Workers from different solved robot regions. The reverse workers
"unsolve" the puzzle with legal reverse pulls; when the forward search and any
reverse search reach the same canonical box layout and robot-reachability
region, the browser stitches both halves into a playable solution.

The regular **Ultimate Search** mode is still available. It uses a multi-worker
portfolio that races complementary strategies such as Push Greedy, Weighted Push
A*, and Push A*. All advanced workers use push-level search, dead-square
pruning, robot-reachability canonicalization, and wall-aware push-distance
heuristics. This is much more appropriate for large Sokoban boards than
expanding every single walking step.

Ultimate modes also race a bounded-memory Push Beam strategy. It uses exact
box-to-goal assignment for all box counts, player-side-aware push pattern maps,
automatic room and chokepoint analysis, constrained-goal packing, robot-region
mobility, and deterministic diversity while retaining only a fixed-width push
frontier. Separate beam quotas preserve promising direct states and strategic
temporary detours. Search priorities count pushes; walking paths are preserved
for replay but do not distract the solver from strategic progress.

On very large boards, direct search uses a sequential memory-bounded restart
portfolio instead of retaining one ever-growing frontier. Restarts alternate
focused, detour, and room-packing profiles with independent deterministic seeds.
Completed games save their best push count as an incumbent upper bound; later
searches prune states whose push lower bound already exceeds that solution.
After the beam attempts, complex boards run fresh-worker incumbent-bounded depth
first searches. These retain a compact transposition table and revisit admissible
branches that a fixed-width beam would permanently discard.

The board analysis is puzzle-independent. It detects articulation gates and
one-entrance rooms, derives farthest-first packing pressure and goal dependencies,
and marks high-traffic packing cells. Hard pruning includes static and player-side
dead squares, label-aware Hall deadlocks, 2x2 and frozen box groups, and sealed
corrals. Globally forced pushes in straight tunnels are collapsed into macros.
Heuristic room ordering affects priority only; it never rejects a state.
Boxes occupying the exterior approach to an unresolved one-entrance room add
congestion pressure, encouraging the solver to clear staging gates before packing.

Small searches remain exhaustive. Complex boards use explicit state and cache
budgets, continuing through independent restarts until a solution is found, the
configured portfolio is exhausted, or you press **Stop**.

Very large puzzles can still hit browser memory limits before the search space
is exhausted. Ultimate Bidirectional uses compact parent records, caps each
bidirectional side on complex boards, bounds worker transposition and memo
tables, and automatically reduces its reverse-worker count. Finished workers
release their frontiers while the sequential restart portfolio continues.

## How to play

Push every box onto its matching goal:

- `X` boxes go on `S` goals.
- Lettered boxes such as `A` go on the matching lowercase goal, such as `a`.
- Boxes can be pushed but cannot be pulled.

Use the arrow keys, WASD, or the on-screen arrow pad to move. Press U to undo
and R to reset. The timer starts on the first legal move.

## Local preview

From the repository root:

```powershell
python -m http.server 8000 --directory docs
```

Then open `http://localhost:8000`.

## Tests

From the repository root:

```powershell
node --test docs/solver-worker.test.js docs/path-validation.test.js
```

## GitHub Pages deployment

1. Push the repository to GitHub.
2. Open **Settings -> Pages**.
3. Choose **GitHub Actions** as the source.
4. Push to `main` or manually run the included Pages workflow.

The `.github/workflows/pages.yml` workflow publishes this `docs/` directory.
No build command or package installation is required.
