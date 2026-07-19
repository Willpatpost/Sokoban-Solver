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
focused, detour, and room-milestone profiles with independent deterministic
seeds. The milestone beam reserves capacity for distinct opening push sequences
and room-crossing histories, including moves that temporarily worsen the raw
box-to-goal estimate.
Completed games save their best push count as an incumbent upper bound; later
searches prune states whose push lower bound already exceeds that solution.
After the beam attempts, complex boards run fresh-worker limited-discrepancy
depth-first searches. Early contours explore only the top-ranked continuation;
later contours admit progressively less obvious choices. This prevents every
restart from exhausting its budget below one attractive branch while retaining
a compact transposition table. A memory-light push IDA* engine is also available
for admissible push-bound contours.

Very large boards also use box-run macros. A local search follows several legal
pushes of one box, including turns, and returns only diverse stopping points to
the global beam. This lets the solver evaluate temporary setup sequences as one
strategic action. Low-distance macro states are handed to several bounded
endgame searches with different packing priorities.
Incomplete macro workers receive a limited push allowance above a known
incumbent, making it possible to discover an easier non-optimal solution before
the exact workers prove or improve the incumbent.

When topology analysis finds a one-entrance room containing more boxes of a
label than the room can ultimately accept, a dedicated evacuation worker treats
moving the surplus through the gate as its first subgoal. After the surplus is
clear, the ordinary assignment and packing scores take over. This objective is
derived from room contents and goals, without puzzle-specific moves or coordinates.

Incomplete forward workers publish replayable box-layout checkpoints. A phase
checkpoint starts a milestone-conditioned reverse worker from the solved layouts;
it discards reverse states that no box assignment can reach from that checkpoint.
The reverse worker publishes stratified landmarks, and bounded bridge workers try
to join compatible checkpoint/landmark pairs. A successful bridge is stitched to
the forward prefix and reverse suffix and replay-validated before it is accepted.
All milestones, targets, and worker assignments are derived from the loaded board.

The board analysis is puzzle-independent. It detects articulation gates and
one-entrance rooms, derives farthest-first packing pressure and goal dependencies,
and marks high-traffic packing cells. Hard pruning includes static and player-side
dead squares, label-aware Hall deadlocks, 2x2 and frozen box groups, and sealed
corrals. Globally forced pushes in straight tunnels are collapsed into macros.
Heuristic room ordering affects priority only; it never rejects a state.
Boxes occupying the exterior approach to an unresolved one-entrance room add
congestion pressure, encouraging the solver to clear staging gates before packing.

Small searches remain exhaustive. Complex boards use explicit state and cache
budgets. After the bounded heuristic portfolio finishes, an exact push-IDA*
worker restarts with geometrically increasing state budgets. It continues until
it finds a solution, proves the state space unsolvable, or you press **Stop**;
there is no fixed push-depth ceiling when no learned incumbent is available.

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
