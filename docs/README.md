# Sokomind Web

The web edition is a responsive, dependency-free browser game ready for GitHub
Pages. It includes five levels from Ultra Tiny through Huge, keyboard play,
undo/reset, a first-move timer, hints, and animated solving in a background Web
Worker.

The move-history panel records every direction, marks pushes, follows
undo/reset, and can copy the complete numbered sequence for sharing.

The search-log panel records the solver's analysis and decisions separately
from player moves. It includes the derived phase plan, worker configuration,
state budgets, expansion rates, heuristic improvements, frontier sizes,
checkpoints, landmarks, bridge handoffs, completions, errors, and 30-second
worker heartbeats. The complete log can be copied for performance comparisons.
Completion entries include elapsed time and an explicit termination reason plus
generated-state, peak-frontier, compaction, and retained-state telemetry when the
worker supplies it. The on-screen field renders the newest 1,500 entries, while
Copy preserves the complete in-memory log for long-run analysis.
Direct-search completions also report compiled-graph construction, dense-board
index construction, heuristic, robot-reachability, cache-hit, push-generation,
and pruning measurements. Robot flood fills and forward push generation use
immutable integer cell IDs and typed arrays while retaining string coordinates
at module boundaries for compatible logs, checkpoints, and replay paths.
Canonical box layouts and robot regions use collision-free dense base-36 keys.
Immutable box-array identities cache those keys so heuristic, deadlock,
transposition, and diversity scoring do not repeatedly sort and stringify the
same layout. Search telemetry includes signature calls, cache hits, generated
characters, and construction time.

Ultimate Bidirectional's planning worker also returns a clone-safe prepared-board
seed. Subsequent portfolio workers reuse its immutable geometry, topology,
reverse-distance tables, dense indices, and compiled single-box graph while
creating private heuristic, deadlock, and signature caches. Workers verify the
exact board contents and schema before reuse and rebuild normally on a mismatch.
This uses standard structured cloning because shared memory requires
cross-origin isolation headers that GitHub Pages does not reliably provide.
The adjacent `{ }` control copies the complete run as newline-delimited JSON.
Each event includes a schema version, run ID, sequence number, timestamp, elapsed
milliseconds, solver build, level, category, message, and typed statistics.

The default solver is **Ultimate Bidirectional**, an experimental Sokoban-aware
search. It starts one forward Web Worker from the current board and one or more
reverse Web Workers from different solved robot regions. The reverse workers
"unsolve" the puzzle with legal reverse pulls; when the forward search and any
reverse search reach the same canonical box layout and robot-reachability
region, the browser stitches both halves into a playable solution.

Before launching that portfolio, a planning worker reads the puzzle and returns
a serializable analysis report. It measures initial push branching and assignment
distance; detects articulation gates, tunnels, gated goal rooms, packing
dependencies, and surplus room boxes; assigns a difficulty band; and recommends
worker counts, beam widths, state budgets, macros, and ordered strategic phases.
The browser director builds the portfolio from this report and records the
rationale in the search log.

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
Milestone landmark generations supersede queued opening bridges instead of sharing
a lifetime quota with them. Bridge candidates are ranked globally, and incompatible
targets do not consume a campaign's viable-search quota. New packing checkpoints
start fresh campaigns. Promising incomplete bridges publish replayable checkpoints
for up to two bounded continuations. Continuations must demonstrate efficient target
progress or reach a credible near-target layout. Each checkpoint and landmark
generation has circuit breakers for incompatible probes, productive workers, visited
states, and cumulative worker time. Bridge exploration uses one execution lane and
cannot replenish indefinitely. The director also retires opening workers once
a packing checkpoint makes their phase stale, while bounded exact handoffs run only
a small contour around each checkpoint.

The board analysis is puzzle-independent. It detects articulation gates and
one-entrance rooms, derives farthest-first packing pressure and goal dependencies,
and marks high-traffic packing cells. Hard pruning includes static and player-side
dead squares, label-aware Hall deadlocks, 2x2 and frozen box groups, and sealed
corrals. Globally forced pushes in straight tunnels are collapsed into macros.
Heuristic room ordering affects priority only; it never rejects a state.
Boxes occupying the exterior approach to an unresolved one-entrance room add
congestion pressure, encouraging the solver to clear staging gates before packing.

Small searches remain exhaustive. Complex boards use explicit state and cache
budgets. After the bounded heuristic portfolio finishes, persistent push-IDA*
workers hash-partition the contour at a shallow push depth. Each shard keeps working
instead of restarting from the root at geometrically increasing state budgets. The
union of the shards covers the contour, and an unsolvability proof is accepted only
after every shard exhausts its partition. Search continues until it finds a solution,
proves the state space unsolvable, or you press **Stop**; there is no fixed push-depth
ceiling when no learned incumbent is available.
Required phase work is tracked separately from opportunistic landmark bridges, so
bridge churn cannot postpone exact search. When required handoffs finish, pending
and active bridges are retired and remaining browser capacity transitions to the
first-solution portfolio.

The first-solution portfolio now separates discovery from proof. Checkpoints are
ranked by accumulated pushes plus their admissible remaining estimate, while phase
diversity prevents every worker from starting at a nearly identical packing layout.
On capable browsers, two cost-aware guided beams search from the best distinct
checkpoints while one persistent exact worker continues the admissible contour.
Packing checkpoints retire active bridge work and extreme puzzles run only the two
best local exact handoffs, allowing the anytime workers to begin earlier. No stored
solution path or puzzle-specific coordinate is used.

Persistent exact progress reports generated successors, threshold and incumbent-bound
prunes, corral and cycle prunes, transposition hits and evictions, shard acceptance,
shard rejection, maximum depth, and the next known contour threshold. Exact and
anytime progress is sampled every 25,000 states to reduce search-log rendering cost.

Very large puzzles can still hit browser memory limits before the search space
is exhausted. Ultimate Bidirectional uses compact parent records, caps each
bidirectional side on complex boards, bounds worker transposition and memo
tables, and automatically reduces its reverse-worker count. A priority work queue
keeps up to two direct-search lanes occupied when the browser reports enough hardware,
while alternating bridge and exact-handoff work so one strategy cannot monopolize
both lanes. Direct capacity expands when the opening forward and reverse workers
finish, allowing later phases to reuse those processor slots. Finished workers
release their frontiers while queued work continues.
Bidirectional heaps compact to their best 40,000 states when they grow past twice
that size. A worker that produces no message for two minutes is terminated, logged,
and replaced by the next portfolio assignment so a wedged worker cannot stall the
director indefinitely.

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
node --test docs/solver-worker.test.js docs/path-validation.test.js docs/director-policy.test.js docs/keyboard-policy.test.js
node --test docs/pruning-differential.test.js bench/evaluator.test.js bench/generated-cases.test.js bench/benchmark.test.js
node bench/benchmark.js --suite smoke
node bench/benchmark.js --suite validation
```

## Benchmarking

The repository includes a headless benchmark harness for algorithm tuning and
AlphaEvolve-style optimization:

```powershell
node bench/benchmark.js --suite alpha --jsonl
```

Each benchmark case runs in an isolated child process, validates any returned
solution by replaying it against the puzzle rules, and emits a single scalar
`totalScore` plus per-case timing, visited states, pushes, estimates, checkpoints,
and cutoff reasons. Invalid returned paths fail the benchmark instead of earning
partial credit. Unsolved partial credit is based on replay-valid checkpoints and
a fixed harness-owned typed push-distance evaluator, not the solver's reported
estimate. The `validation` suite adds deterministic mirrored, rotated, relabeled,
and premature-goal cases to reduce tuning against only the built-in layouts.

## GitHub Pages deployment

1. Push the repository to GitHub.
2. Open **Settings -> Pages**.
3. Choose **GitHub Actions** as the source.
4. Push to `main` or manually run the included Pages workflow.

The `.github/workflows/pages.yml` workflow publishes this `docs/` directory.
No build command or package installation is required.
