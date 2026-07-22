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

Run the deterministic validation suite of mirrored, rotated, relabeled, and
premature-goal cases:

```powershell
node bench/benchmark.js --suite validation --jsonl
```

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
- `totalScore`: single scalar objective for evolutionary search.
- `cases`: per-case timing, visited states, solution length, pushes, estimates,
  replay-validated checkpoint evaluation, cutoff reason, and compact progress
  metadata. Unsolved partial credit uses only the harness's fixed evaluator;
  solver-reported estimates remain telemetry and do not affect the score.

For AlphaEvolve, optimize `totalScore` while treating any `valid: false` or
non-zero `errors` as a hard rejection. The benchmark intentionally rewards a
curriculum: easy levels must remain correct, while hard and huge cases carry
larger weights.
