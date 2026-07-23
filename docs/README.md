# Sokomind Web

The web edition is a responsive, dependency-free browser game ready for GitHub
Pages. It includes five levels from Ultra Tiny through Huge, keyboard play,
undo/reset, a first-move timer, hints, and animated solving in a background Web
Worker.

The browser runtime is split by responsibility without requiring a build tool:
game rules, telemetry formatting, page UI, worker orchestration, solver rules and
heuristics, search algorithms, and the Web Worker protocol each have an explicit
module boundary. Pure rule and formatting modules run directly in Node tests, while
the page and worker load the same files as classic scripts.

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
goal-commitment, support-dependency, local-room/corral, typed doorway-flow, heap,
worker startup/wall-clock/termination, and pruning measurements. Unsupported
browser heap APIs are omitted rather than displayed as zero. Robot flood fills
and forward push generation use
immutable integer cell IDs and typed arrays while retaining string coordinates
at module boundaries for compatible logs, checkpoints, and replay paths.
Canonical box layouts retain collision-free dense base-36 keys for memo tables
and worker boundaries. Hot search transposition maps instead use collision-free
packed `BigInt` identities that combine sorted typed-box cell tokens with the
exact robot cell or canonical reachable-region ID. Immutable box arrays cache
their packed identity, and diversity sharding hashes that dense value directly.
Search telemetry reports both string-signature and packed-identity activity.
Successor layouts also retain a weak parent hint for the one box that moved.
When the parent's exact Hungarian assignment is cached, the heuristic reuses all
unchanged cost rows and repairs the matching with one O(n-squared) augmenting-path
update instead of rebuilding it in O(n-cubed). Repair is enabled for label groups
of at least five boxes, where profiling supports the tradeoff; smaller groups keep
the simpler full calculation. Deterministic matrix tests compare
the repair against full recomputation, and telemetry reports repairs, fallbacks,
and reused rows.

Bounded goal rooms can also compile relaxed multi-box pattern tables for groups
of two to four labels whose complete goal sets belong to that room. The reverse
table keeps walls, typed goals, box collisions, and required push support squares,
while removing every unrelated box and granting the robot access everywhere.
Its exact pattern distance is therefore admissible. The heuristic replaces the
same labels' Hungarian contribution with the stronger value instead of adding
both, and combines only label-disjoint room patterns. Tables stop after 12,000
states; discovered distances remain exact, while missing cutoff entries leave the
ordinary assignment bound unchanged.

Unique-label box pairs whose shortest push corridors overlap a tunnel or
articulation receive a second bounded check. A reverse two-box table removes all
other boxes and robot-connectivity constraints, so any interaction cost it proves
is also admissible. Pair tables are limited to nearby conflicts with at most 18
independent pushes and 4,000 reverse states. Room and pair improvements enter one
label-disjoint selection—an exact maximum for up to 20 involved labels—preventing
a box's unavoidable detour from being counted twice. Cutoffs and shared-label
assignments retain the Hungarian bound.

Ultimate Bidirectional's planning worker also returns a clone-safe prepared-board
seed. Subsequent portfolio workers reuse its immutable geometry, topology,
reverse-distance tables, dense indices, and compiled single-box graph while
creating private heuristic, deadlock, and signature caches. Workers verify the
exact board contents and schema before reuse and rebuild normally on a mismatch.
Compiled room-pattern and pair-conflict tables are included in that seed so
portfolio workers do not repeat their bounded reverse enumeration.
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

Each oversized beam layer also reserves a profile-dependent feature archive.
Coarse joint cells distinguish heuristic slack, room and evacuation pressure,
packing progress, doorway state, support dependencies, local-room guidance, and
robot mobility. Candidates are drawn round-robin from those cells before the
remaining slots use the established heuristic bands and push-class diversity.
This keeps strategically different states alive even when their scalar scores are
worse. The archive is bounded by the existing beam width and can be disabled for
diagnostic comparisons; exact searches are unchanged.

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
cannot replenish indefinitely. A successful evacuation retires pending opening work
and releases active opening workers from the required portfolio; packing repeats the
release defensively. Bounded exact handoffs run only a small contour around each
checkpoint.

The board analysis is puzzle-independent. It detects articulation gates and
one-entrance rooms, derives farthest-first packing pressure and goal dependencies,
and marks high-traffic packing cells. Hard pruning includes static and player-side
dead squares, label-aware Hall deadlocks, 2x2 blocks, conservative wall-ended
closed diagonals, frozen box groups, and sealed corrals. Globally forced pushes in
straight tunnels are collapsed into macros.
Small non-branching typed-box neighborhoods additionally use a cached relaxed
pattern search. The robot may stand on every legal support square and boxes may
escape through the pattern boundary, so only a pattern unsalvageable under those
more generous rules is rejected. Static windows are compiled once per center;
cutoffs and larger or branching neighborhoods retain the state.
Heuristic room ordering affects priority only; it never rejects a state.
Boxes occupying the exterior approach to an unresolved one-entrance room add
congestion pressure, encouraging the solver to clear staging gates before packing.

Goal packing uses a conservative commitment oracle instead of rewarding every
currently matched box equally. A placement is temporary when fixing it would
block pending room dependencies, occupy a required gate, or destroy the remaining
typed perfect matching and its static or dynamic support routes. Movable placements
that pass those checks are conditional and receive a reduced ordering reward. A
matched box is proven when it is statically immovable or belongs to an exactly
completed one-entry room whose imports, exports, gate, staging, matching, and support
requirements are resolved. Beam, bounded DFS, and IDA* keep proven boxes fixed;
baseline search and gameplay retain ordinary legal movement. Commitment is cached by
box layout and robot region, and macros re-evaluate it after each push.

Advanced forward searches also build a bounded, dynamic support-square dependency
graph for each expanded robot region and box layout. Goal-directed push gradients
identify useful box destinations and their required robot standing squares. When a
standing square is inaccessible, a minimum-blocker route records the boxes that must
move first; candidate pushes that clear those prerequisites are ordered earlier,
while pushes that occupy demanded staging sides are ordered later. The graph is
cached and instrumented, and remains an ordering signal rather than a pruning rule.

Small gated goal rooms and genuine inaccessible corrals use a shared cached exact
push-state search. Room entries include nearby staging cells and track typed imports,
exports, doorway occupancy, restored doorway states, packed goals, and viable boundary
configurations. Corral entries search for the shortest way to reopen robot access or
resolve the enclosed goals. Completed local searches expose their optimal first pushes
to beam, discrepancy, and IDA* ordering. Exhausted, oversized, inaccessible, and
budget-cutoff abstractions are reported but never used as global hard-pruning proofs.
Only an exactly completed room can support proven-box locking; corral-opening hints
receive lower confidence and remain ordering-only.

Bounded one-entrance goal rooms also compile a reusable reverse packing table from
the fully packed typed-goal arrangement. Reverse-legal pushes retain the robot's
reachable-region identity, so table hits provide an exact remaining push count and
all optimal next packing pushes. Tables are limited by room, domain, box, and state
budgets; unsupported or cutoff rooms fall back to the forward local search. A missing
entry is diagnostic for the bounded doorway abstraction and is never treated as an
unconditional global deadlock, since moving farther outside the abstraction may help.

One-entrance room topology now compiles doorway lanes and nearby interior/exterior
staging sets. A cached typed-flow analysis compares each room's current labels with
its target labels, records required imports and exports, checks whether the geometry
supports each direction, measures open staging capacity and dynamically ready lanes,
and reports blocked gates or capacity contradictions. Push ordering favors the right
label moving through the gate in the required direction and discourages consuming
needed staging cells. These constraints remain diagnostic ordering pressure rather
than hard pruning.

Small searches remain exhaustive. Complex boards use explicit state and cache
budgets. After the bounded heuristic portfolio finishes, persistent push-IDA*
workers hash-partition the contour at a shallow push depth. IDA* uses an explicit
traversal stack rather than JavaScript recursion. Production proof workers yield
after bounded work slices and save a versioned, board-hashed checkpoint containing
their contour, shard, stack, and bounded recent performance cache. A compatible
checkpoint resumes after a worker restart or page reload; incompatible board,
build, bound, and shard data is rejected. **Clear Saved Search** explicitly discards
the current puzzle's durable proof state.

The union of the shards covers the contour, and an unsolvability proof is accepted
only after every shard exhausts its partition. Search continues until it finds a
solution, proves the state space unsolvable, reports a platform failure, or you
press **Stop**; there is no fixed push-depth ceiling when no learned incumbent is
available.
Required phase work is tracked separately from opportunistic landmark bridges, so
bridge churn cannot postpone exact search. When required handoffs finish, pending
and active bridges are retired and remaining browser capacity transitions to the
first-solution portfolio.

The first-solution portfolio now separates discovery from proof. Checkpoints are
ranked by accumulated pushes plus their admissible remaining estimate, while phase
diversity prevents every worker from starting at a nearly identical packing layout.
On capable browsers, cost-aware guided beams search from the best distinct
checkpoints while one persistent exact worker continues the admissible contour.
Packing checkpoints retire active bridge work and extreme puzzles run only the two
best local exact handoffs, allowing the anytime workers to begin earlier. No stored
solution path or puzzle-specific coordinate is used.

Guided beam and bounded DFS workers publish progress on both state-count and
elapsed-time intervals, so expensive states cannot make a productive worker appear
dead merely because it has not reached a large state-reporting threshold. If a
guided beam nevertheless remains silent for the full watchdog interval, the
director launches one reduced-cost recovery from the same general state. Sequence
macros and nested endgame probes are disabled, width and state count are bounded,
and a second watchdog failure is final rather than an unbounded restart loop.

Persistent exact progress reports generated successors, threshold and incumbent-bound
prunes, corral and cycle prunes, transposition hits and evictions, shard acceptance,
shard rejection, maximum depth, and the next known contour threshold. Exact progress
is sampled by state count while guided workers also send bounded elapsed-time
liveness reports. Exact transposition capacity is divided from a device-memory-aware
total budget: a lone proof shard can retain more states, while parallel shards remain
collectively bounded.

Every public run ends with one of five meanings: `solved`,
`proven-unsolvable`, `cutoff`, `cancelled`, or `failed`, plus a machine-readable
reason. Only complete exact exhaustion may report `proven-unsolvable`; finite
bounds and guided budgets report `cutoff`. Browser and Python solvers independently
replay a candidate path before exposing `solved`. Repeated exact-worker failures
resume from the last compatible proof checkpoint and stop with `failed` after a
bounded recovery count rather than looping forever.

Exact search separates commitment checks used for proven box locking from richer
child-state signals used only to order equal-bound successors. Proven locking remains
active on every expanded state. Child commitment and doorway scoring is evaluated
only for successors inside the current contour; a productivity gate samples whether
those signals actually change successor order, pauses them after an unproductive
64-state window, and samples again after 512 eligible states. Progress telemetry
reports strategic ordering evaluations, skips, and ordering changes.

Very large puzzles can still hit browser memory or storage limits before the search
space is exhausted; such a platform failure is not an unsolvability proof. Ultimate
Bidirectional uses compact parent records, caps each
bidirectional side on complex boards, bounds worker transposition and memo
tables, and automatically reduces its reverse-worker count. A priority work queue
uses capacity released by completed forward and reverse workers. While evacuation
remains active, direct searches occupy at most two lanes, preserving parallel
discovery without letting opening work crowd out the checkpoint producer. Once
evacuation succeeds, queued opening plans are retired and active opening workers
become opportunistic: they may still find a solution, but they no longer delay
required handoffs or persistent exact search. Only one landmark bridge may be
active. During the exact phase, finished bridge and anytime workers are replaced with distinct
checkpoint-guided profiles whenever eligible checkpoints remain. Each checkpoint is
attempted at most twice with rotating profiles and seeds; the persistent exact shard
is never displaced, and a slot remains idle rather than running an unbounded duplicate.
Finished workers release their frontiers before replacement work begins.
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
node --test docs/solver-worker.test.js docs/solver-worker-protocol.test.js docs/exact-kernel-differential.test.js docs/exact-checkpoint-storage.test.js docs/browser-modules.test.js docs/path-validation.test.js docs/director-policy.test.js docs/keyboard-policy.test.js docs/conformance.test.js
node --test docs/pruning-differential.test.js bench/evaluator.test.js bench/generated-cases.test.js bench/benchmark.test.js bench/conformance.test.js bench/verify-solution.test.js bench/solver-generality.test.js
node bench/verify-solution.js huge docs/optimalForHuge.txt
node bench/benchmark.js --suite smoke
node bench/benchmark.js --suite validation
npm ci
npx playwright install chromium webkit
npm run test:browser
```

The shared conformance fixtures verify that Python, the browser worker, and the
benchmark replay engine agree on the level catalog, symbols, typed and generic
box/goal counts, ragged walls, goal semantics, mechanically legal moves, and
deadlock-pruned search successors.

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
premature-goal, bottleneck, staging-capacity, coupled-room, dependency-cycle, and
multi-gate cases to reduce tuning against only the built-in layouts. Bounded
generated cases carry reviewed push expectations certified by an independent
exact push search; their private seeds never enter solver payloads.

## GitHub Pages deployment

1. Push the repository to GitHub.
2. Open **Settings -> Pages**.
3. Choose **GitHub Actions** as the source.
4. Push to `main` or manually run the included Pages workflow.

The `.github/workflows/pages.yml` workflow publishes this `docs/` directory.
No build command or package installation is required.
