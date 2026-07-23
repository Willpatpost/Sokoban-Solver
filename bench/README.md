# Sokomind Benchmark Harness

This harness runs the browser solver kernels under Node in isolated child
processes. Each case is replay-validated against the Sokoban rules before it can
earn solved credit, so optimization agents cannot win by returning illegal paths.

Run the fast smoke suite:

```powershell
node bench/benchmark.js --suite smoke
```

Run the longer AlphaEvolve-oriented suite:

```powershell
node bench/benchmark.js --suite alpha --jsonl
```

Run the deterministic validation suite of mirrored, rotated, relabeled,
premature-goal, typed-doorway, exact-room, corral-family, and separately seeded
bottleneck, staging-capacity, coupled-room-ordering, dependency-cycle, multi-gate,
and three- and four-box cases:

```powershell
node bench/benchmark.js --suite validation --jsonl
```

Gate the reviewed smoke and validation counters and heap envelopes:

```powershell
node bench/performance-gate.js
```

Profile the exact-assignment one-row repair crossover in both runtimes:

```powershell
node bench/profile-assignment.js
python bench/profile_assignment.py
```

The reviewed result is stored in `assignment-crossover.json`. JavaScript repairs
groups of at least three boxes and Python repairs groups of at least five; below
those thresholds a full Hungarian calculation was faster on the reviewed
runtimes. An exact one-row update still needs an augmenting path across the
matching and therefore reads O(n-squared) cost information in the worst case;
it cannot generally be reduced to O(n).

Profile construction, structured cloning, and hydration of the Huge prepared
board:

```powershell
node bench/profile-prepared-board.js
```

`prepared-board-profile.json` records the reviewed machine result. It is
diagnostic rather than a timing gate because worker startup and structured-clone
latency vary by browser and machine.

`local-reasoning-baseline.json` records the deterministic proof ceilings for
local room/corral and deadlock-pattern enumeration. Unit tests require production
constants to match it; exceeding a ceiling returns `oversized` or retains the
state rather than turning an incomplete search into a rejection.

`performance-baselines.json` records visited/generated/retained states, frontier
peaks, transposition evictions, and isolated-process heap peaks. Deterministic
counters use narrow reviewed tolerances. Heap has a separate wider tolerance,
and elapsed time is reported but never gated. The Huge suite runs only in the
scheduled or manually dispatched `Huge Performance` workflow.

The procedural families are separate from the built-in levels. Their private
seeds never enter solver payloads, and an independent exact push-state search in
`generated-cases.test.js` checks every reviewed solved/push expectation.

Run the expensive Huge-focused suite when the cluster allocation is intended for
that purpose:

```powershell
node bench/benchmark.js --suite huge --jsonl
```

Run one level/algorithm pair:

```powershell
node bench/benchmark.js --level huge --algorithm push-beam --max-visited 250000 --timeout-ms 60000
```

The final JSON object contains:

- `schemaVersion`: currently `2`; version 2 adds harness-owned checkpoint
  evaluation and changes unsolved partial-credit semantics.
- `solved`: replay-valid solutions found.
- `valid`: false if any returned path failed replay validation.
- `totalVisited`: total solver states visited across child processes.
- `memorySupportedCases` and `peakHeapBytes`: suite-wide heap coverage and peak.
- `totalChildProcessMs` and `totalShutdownMs`: aggregate isolated-runner lifecycle
  and post-result process shutdown time.
- `totalScore`: single scalar objective for evolutionary search.
- `cases`: per-case timing, visited states, solution length, pushes, estimates,
  replay-validated checkpoint evaluation, cutoff reason, and compact progress
  metadata. Unsolved partial credit uses only the harness's fixed evaluator;
  solver-reported estimates remain telemetry and do not affect the score.
- `performance`: worker-owned hot-path timings, call counts, cache hits, compiled
  graph size, dense-board build size/time, generated push candidates, retained
  pushes, compact-signature construction/cache behavior, and deadlock prunes.
  Dense-layout derivations, occupancy-bitset copies, typed-goal reverse-table
  hits, assignment repairs, prepared-board hydration, and safe-fallback counts
  are included when applicable.
- Each case also includes `runnerLifecycle` for worker loading, search,
  validation, total time, and heap delta, plus `processLifecycle` for first
  output, result delivery, shutdown, and total child-process time. Heap fields
  explicitly report unsupported hosts instead of substituting zero.

Memory records use one versioned shape with an explicit support flag and source.
Neither browsers nor ordinary Node processes guarantee a collection before a
sample, so `gcControlled` is false and heap values are treated as envelopes,
not exact retained-object measurements.

For AlphaEvolve, optimize `totalScore` while treating any `valid: false` or
non-zero `errors` as a hard rejection. The benchmark intentionally rewards a
curriculum: easy levels must remain correct, while hard and huge cases carry
larger weights.
