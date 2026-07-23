Sokomind forward roadmap
=========================

Mission
-------

Make Sokomind produce a good solution quickly, improve it for as long as the user
allows, and retain a complete exact path toward an optimality or unsolvability
proof. Every decision must be derived from the supplied puzzle. Built-in names,
coordinates, saved routes, and the diagnostic Huge solution are forbidden inputs
to production search.

Current reference point
-----------------------

Build 2026-07-23.33 solves Huge with the released structural planner in 3,752
visited states, 19,689 generated candidates, 330 pushes, and 1,306 player moves.
The measured run takes about 85-87 seconds and succeeds with a 256 MB V8 heap
ceiling. The diagnostic solution demonstrates that at least 250 pushes and 640
moves are possible; it is benchmark evidence, not solver input and not proof of
optimality.

Roadmap rules
-------------

- The seven highest-priority improvements are Tasks 1-7.
- Exact search must remain complete. A discovery rule may reject a branch only
  inside an explicitly incomplete lane; production hard pruning requires a proof
  and independent differential tests.
- Deliver the first replay-valid solution immediately. Later optimization must
  never delay or invalidate it.
- Prefer generated, transformed, and independently certified puzzles over
  assertions tied only to Huge.
- Measure deterministic states, generated candidates, retained records, cache
  occupancy, and solution quality separately from machine-sensitive elapsed time.
- A task is finished only when its acceptance gate is satisfied. Partial
  foundations are noted solely to prevent duplicated work.


Sprint 0 - Cleaning, organizing, and refactoring
================================================

Status: Complete in build 2026-07-23.33.

This repository-hygiene sprint precedes the numbered solver campaign without
changing its priorities.

Delivered:

- Moved the desktop guide and forward roadmap out of the repository root and
  linked them from the primary README.
- Standardized the Python package directory as lowercase `searches/` and updated
  imports, commands, static checks, coverage configuration, CI, and architecture
  documentation.
- Documented the repository layout and kept only conventional entry points,
  manifests, and tool-discovery configuration at the root.
- Consolidated ignored Python, browser-test, coverage, virtual-environment, and
  log artifacts; removed generated local artifacts from the working tree.
- Preserved the GitHub Pages layout, Python script entry points, cross-runtime
  fixtures, and all existing test and benchmark contracts.

Completion gate:

- No tracked code or documentation refers to the former paths.
- A clean test run creates only explicitly ignored artifacts.
- JavaScript, Python, browser, formatting, typing, coverage, build, and benchmark
  gates pass after the case-sensitive rename.


Sprint 1 - Faster structural first solutions
============================================

Goal: preserve the reliable structural route while eliminating work before it
becomes a full macro candidate and reducing contention around the most productive
worker.

1. Apply structural rejection during macro expansion
   Status: Partial; the current guards primarily evaluate macro endpoints.

   Plan:
   - Build an incremental macro context containing the active assignment,
     doorway phase, staging obligations, packing dependencies, exporter egress,
     box mobility, and unsolved-goal access.
   - Evaluate the context after every intermediate push and stop a macro as soon
     as it creates a discovery-lane contradiction that cannot be repaired within
     that macro.
   - Keep proven deadlocks separate from discovery-only rejection reasons.
     Record each category independently in telemetry.
   - Reuse parent analysis where one pushed box is the only changed element;
     avoid recomputing whole-board maps at every intermediate state.
   - Add unrelated doorway, temporary-displacement, and packing-order fixtures
     that require both early rejection and deliberate temporary worsening.

   Acceptance gate:
   - Base and transformed Huge still solve with replay-valid paths under released
     limits.
   - Exact-search outcomes remain identical to the independent oracle.
   - Macro intermediate states and dense-layout derivations fall materially
     without increasing retained structural states or peak memory.

2. Make macro effort adaptive
   Status: Planned; current exploration limits are fixed.

   Plan:
   - Estimate local ambiguity from legal push directions, competing targets,
     blocker routes, doorway conflicts, and structural score dispersion.
   - Use small lookahead for forced or low-ambiguity motion and reserve the full
     search budget for doorway crossings, staging decisions, and packing order.
   - Widen a macro only after its cheaper pass fails to produce a structurally
     distinct endpoint; contract it again after decisive progress.
   - Preserve multiple destinations when scores are tied instead of allowing
     direction iteration order to choose the only survivor.
   - Report budget tier, widening reason, endpoints retained, and useful-result
     rate so thresholds can be tuned from evidence.

   Acceptance gate:
   - The adaptive policy solves every case solved by the fixed 96-state policy in
     the reviewed structural suite.
   - Easy and forced macro families perform less work, while ambiguous families
     retain the required alternatives.
   - Huge improves in wall time or generated intermediate work without regressing
     its deterministic top-level search counters.

3. Give the structural worker resource priority
   Status: Partial; it is scheduled first but still competes with the full
   Ultimate portfolio.

   Plan:
   - Give structural discovery a short head start or exclusive high-priority slot
     on puzzles where analysis predicts a strong doorway/packing plan.
   - Scale concurrent discovery and reverse workers from hardware concurrency,
     supported memory, structural progress, and measured marginal yield.
   - Start proof work by a fixed deadline, but cancel or defer redundant guided
     lanes while the structural worker is making milestone progress.
   - Replace fixed profile counts with explicit first-solution, improvement, and
     proof capacity reservations.
   - Add deterministic browser campaigns for head-start success, structural
     plateau, worker failure, low-memory devices, and exact fallback.

   Acceptance gate:
   - Real-browser campaigns show lower time to first solution and lower aggregate
     worker memory on complex puzzles.
   - Structural work cannot starve exact proof, and opportunistic work cannot
     delay a required handoff.
   - Stop, reset, worker failure, and watchdog recovery remain bounded and tested.

Sprint 1 completion gate:
Huge and transformed variants retain their reliable solution, first-solution wall
time improves on the reviewed machine, and the reduction is explained by lower
intermediate work and/or lower worker contention rather than weaker validation.


Sprint 2 - Anytime solution quality and proof acceleration
===========================================================

Goal: turn the first structural solution into an incumbent that immediately helps
find better solutions and constrains complete search.

4. Continue structural search after the first solution
   Status: Planned.

   Plan:
   - Publish and animate the first replay-valid solution immediately, then move
     the run into an explicit improving state.
   - Continue from retained structural elites with the incumbent push count as an
     upper bound.
   - Explore alternative compatible assignments, doorway waves, packing orders,
     and deterministic diversity seeds instead of repeating the winning plan.
   - Maintain a Pareto set ordered by pushes first, then player moves, memory,
     and discovery time; emit only replay-validated improvements.
   - Persist the incumbent and enough bounded planner state to resume improvement
     after a worker restart or page reload.

   Acceptance gate:
   - First-solution latency is unchanged or better.
   - Reviewed multi-plan puzzles receive monotonically improving incumbents.
   - Huge produces a solution below 330 pushes within a reviewed improvement
     budget, with every incumbent independently replayed.

5. Rewrite completed solutions with exact local windows
   Status: Planned.

   Plan:
   - Partition a solution at stable structural milestones such as completed
     exports, imports, room packing, and goal commitments.
   - Re-solve bounded windows between fixed boundary states using exact push
     search, expanding a window only when its state-space estimate is safe.
   - Optimize pushes first and player moves second; reconstruct shortest walking
     paths between the retained push actions.
   - Iterate over overlapping windows until a full pass produces no improvement,
     while preventing oscillation with canonical boundary identities.
   - Reject a replacement unless the entire stitched path replays to the same or
     a solved final state.

   Acceptance gate:
   - Rewriting never worsens pushes or moves and never changes puzzle semantics.
   - Authored detour fixtures are reduced to independently verified local optima.
   - Huge improves measurably beyond the raw structural solution within a bounded
     post-processing time and memory budget.

6. Feed every incumbent into exact search
   Status: Partial; exact algorithms accept bounds, but the full anytime director
   does not yet exploit a live improving incumbent end to end.

   Plan:
   - Start or tighten exact IDA*/A* with the best replay-valid push bound as soon
     as an incumbent is available.
   - Propagate bound reductions safely to active shards and persisted checkpoints;
     restart only work whose proof contract is invalidated by the tighter bound.
   - Separate “best known solution,” admissible lower bound, and proven optimum
     in the UI, logs, and terminal result.
   - Prioritize contours and shards by expected bound-closing value while
     preserving complete partition coverage.
   - Continue exact proof after discovery workers retire, subject to explicit
     user stop and platform failure.

   Acceptance gate:
   - Differential tests prove that live bound tightening cannot remove an optimal
     solution or create a false unsolvability/optimality claim.
   - Exact search visits fewer states on reviewed incumbent-sensitive families.
   - A completed proof reports an independently verified optimum; an incomplete
     run reports an honest remaining gap.

Sprint 2 completion gate:
Sokomind returns the first solution promptly, improves it monotonically, and uses
each improvement to reduce exact proof work without overstating optimality.


Sprint 3 - Compact and nonredundant search
==========================================

Goal: reduce memory and equivalent work after the first-solution and improvement
flows are stable.

7. Compact macro-state and path storage
   Status: Partial; dense identities exist, but macro candidates still retain
   repeated arrays, objects, and path segments.

   Plan:
   - Store box layouts as packed immutable tokens with one moved-box delta where
     practical.
   - Replace object parent chains with arena indices and shared encoded move
     segments.
   - Retain full robot paths only for surviving candidates and reconstruct
     discarded walking detail on demand.
   - Give discovery, goal-access, assignment, and macro caches explicit phase
     lifetimes and independent caps.
   - Report live versus cumulative allocations, arena occupancy, cache occupancy,
     and compaction cost.

   Acceptance gate:
   - Search identities, replay paths, and deterministic outcomes are unchanged.
   - Huge remains below the 256 MB ceiling and demonstrates a reviewed reduction
     from the approximately 218 MB measured peak.
   - Compaction does not increase first-solution time or hide retained memory.

8. Finish adaptive feature-space queues
   Status: Partial; beams preserve structural and heuristic elites, but quotas
   and identities are mostly fixed.

   Plan:
   - Track the yield of room-flow, doorway, access, packing, mobility, and
     assignment feature cells across depths and restarts.
   - Adapt cell boundaries and quotas only from prior-window evidence.
   - Preserve productive elites while retiring duplicate identities and stagnant
     cells.
   - Add generated feature-conflict families where the best heuristic state and
     the necessary structural state differ.
   - Port the stable, useful subset to Python or document why it remains
     browser-only.

   Acceptance gate:
   - Feature-conflict families retain required detours with fewer states than the
     fixed policy.
   - Adaptive queues do not regress small exact outcomes or Huge reliability.

9. Add proof-backed partial-order reduction
   Status: Planned.

   Plan:
   - Define push independence using affected boxes, support squares, robot access,
     doorway phases, commitments, and macro side effects.
   - Canonicalize only interleavings whose actions commute and whose intermediate
     states preserve the same legal continuation set.
   - Disable reduction whenever the certificate is incomplete.
   - Exhaustively compare reduced and unreduced search on small multi-box boards.
   - Measure eliminated interleavings separately from ordinary transposition hits.

   Acceptance gate:
   - Solvability and optimal push counts match the independent oracle.
   - Reviewed independent-action families show a deterministic state reduction.

10. Complete symmetry reduction
    Status: Partial; structural discovery canonicalizes board orientation, but
    state-space symmetries and interchangeable boxes are not fully reduced.

    Plan:
    - Detect automorphisms that preserve walls, goals, labels, and movement rules.
    - Canonicalize symmetric box layouts without merging distinct typed boxes or
      robot-reachability classes.
    - Combine symmetry keys with transposition and partial-order keys in one
      documented identity contract.
    - Retain the current rotation/reflection path and checkpoint restoration.
    - Validate mirrored, rotated, relabeled, and asymmetric counterexamples.

    Acceptance gate:
    - Symmetric searches retain exact outcomes and show fewer unique states.
    - Asymmetric or typed puzzles are never merged under an invalid symmetry.

Sprint 3 completion gate:
Memory and unique-state work decrease under deterministic gates, and every
reduction has a tested fallback that preserves exact completeness.


Sprint 4 - Exact runtime resilience and observability
=====================================================

Goal: make long proof runs efficient, diagnosable, and resumable after structural
discovery has supplied a useful incumbent.

11. Improve exact transposition retention
    Status: Partial; capacity is bounded, but eviction is insertion-ordered and
    checkpoint retention value is not measured.

    Plan:
    - Separate live table occupancy, cumulative retained entries, checkpoint tail,
      and eviction counts.
    - Compare contour-aware, depth-aware, and recency policies under the same total
      memory budget.
    - Preserve entries that prevent repeated work after checkpoint resume.
    - Adapt capacity only from supported memory pressure signals and retain a
      conservative fixed fallback.
    - Test shard unions and resumed proofs with tiny forced-eviction capacities.

    Acceptance gate:
    - Exact outcomes and shard coverage remain unchanged.
    - Reviewed long contours repeat less work without increasing the memory cap.

12. Add cooperative proof slices and complete lifecycle accounting
    Status: Partial; resumable slices and watchdogs exist, but expensive
    expansions and cancellation summaries remain incomplete.

    Plan:
    - Bound work between cooperative yields even when one expansion performs
      expensive local reasoning.
    - Split worker-ready, checkpoint-load, first-expansion, first-progress, and
      shutdown timings.
    - Emit a release record for every solved, failed, cutoff, cancelled, replaced,
      or watchdog-terminated worker.
    - Include terminal visited/generated counts rather than the previous progress
      sample.
    - Exercise stop and recovery paths in real Chromium and WebKit campaigns.

    Acceptance gate:
    - No productive worker is falsely declared silent.
    - Every started worker has exactly one terminal lifecycle record.
    - Recovery is bounded and cannot loop indefinitely.

13. Produce one compact terminal campaign summary
    Status: Partial; detailed events exist, but user-stopped runs still require
    manual reconstruction.

    Plan:
    - Version the exported search-log schema and normalize all identifiers as
      strings.
    - Summarize the incumbent, best replayable checkpoint, exact contour and gap,
      per-strategy states/time, lifecycle totals, and supported memory.
    - Emit the same summary for solved, proven-unsolvable, cutoff, cancelled, and
      failed runs.
    - Keep the summary bounded regardless of campaign duration.
    - Add fixture-based compatibility tests for old and new log readers.

    Acceptance gate:
    - One terminal record is sufficient to determine what the campaign achieved,
      what remains, and why it ended.

Sprint 4 completion gate:
Long exact campaigns resume efficiently, yield predictably, and explain their
final proof state without reconstructing thousands of log records.


Sprint 5 - Runtime consolidation and intentional parity
=======================================================

Goal: remove implementation and product duplication after the new search contract
is stable.

14. Finish browser runtime modularization
    Status: Partial.

    Plan:
    - Split topology, assignment/heuristic, deadlock proof, structural planning,
      and local exact analysis behind explicit interfaces.
    - Centralize shared rule primitives and remove test-loader duplication.
    - Keep classic-script/Web Worker deployment compatible with GitHub Pages and
      avoid adding a bundler solely for organization.
    - Add dependency-boundary tests that prevent UI, director, and proof logic
      from merging again.

    Acceptance gate:
    - Module ownership is documented and cyclic dependencies are absent.
    - Build, worker loading, browser tests, and deployment remain unchanged.

15. Consolidate public search modes
    Status: Planned.

    Plan:
    - Present a small product contract: Recommended, Quick/anytime, and
      Exact/proof, each with honest guarantees.
    - Keep useful internal algorithms available to the director, benchmarks, and
      an advanced developer interface.
    - Remove duplicate Fast/Portfolio/Ultimate aliases and obsolete compatibility
      launchers after a documented migration period.
    - Use ablations before removing any strategy that contributes unique value.
    - Simplify dispatch, tests, documentation, and telemetry names together.

    Acceptance gate:
    - Every public mode has distinct behavior and accurate guarantees.
    - No removed alias is still referenced by UI, CLI, tests, or documentation.

16. Bring Python to intentional parity
    Status: Partial.

    Plan:
    - Define a browser/Python capability table rather than blindly porting every
      worker-specific feature.
    - Port the stable structural planner, solution validation, compact state,
      anytime incumbent, and selected exact improvements needed by local users.
    - Share conformance and benchmark fixtures wherever language boundaries allow.
    - Remove redundant Python heuristics and aliases superseded by the agreed
      architecture.
    - Document intentionally browser-only orchestration and UI features.

    Acceptance gate:
    - Shared capabilities produce compatible statuses and replay-valid solutions.
    - Intentional differences are explicit, tested, and documented.

Sprint 5 completion gate:
The codebase and public search choices are smaller, each runtime has a deliberate
scope, and no compatibility bloat remains without a documented consumer.


Sprint 6 - Accessible, polished, release-ready product
======================================================

Goal: finish the user-facing solver and repository before adding puzzle creation.

17. Complete screen-reader and reduced-motion support
    Status: Planned.

    Plan:
    - Provide an accessible board description covering robot, boxes, goals, move
      count, completion, and solver state.
    - Scope live announcements so progress does not flood assistive technology.
    - Preserve keyboard focus during search, playback, undo, reset, and dialogs.
    - Honor reduced-motion preferences for transitions and solution playback.
    - Test semantic state and focus behavior in browser integration tests.

    Acceptance gate:
    - The full play/search workflow is usable without visual board inspection or
      animation.

18. Complete desktop and mobile interaction review
    Status: Planned.

    Plan:
    - Refine hierarchy, controls, board sizing, logs, touch input, and long-status
      overflow while preserving the current visual identity.
    - Validate supported breakpoints, orientation changes, zoom, focus visibility,
      and minimum touch targets.
    - Ensure long searches and solution playback do not destabilize layout.
    - Add automated breakpoint coverage and perform a documented visual review.

    Acceptance gate:
    - Supported desktop and mobile layouts have no clipped controls, inaccessible
      actions, or board overflow regressions.

19. Complete release and repository hygiene
    Status: Partial.

    Plan:
    - Add or finalize LICENSE, CONTRIBUTING, support matrix, release notes, and
      browser/Python capability documentation.
    - Verify setup, testing, benchmarking, deployment, build versioning, and
      troubleshooting instructions from a clean checkout.
    - Define a changelog and compatibility policy for puzzles, logs, checkpoints,
      and CLI modes.
    - Ensure all required checks run before deployment with least-privilege
      permissions.

    Acceptance gate:
    - A new contributor can build, test, benchmark, and deploy using only the
      documented steps.

Sprint 6 completion gate:
The solver is accessible, responsive, documented, and releasable on every
supported platform.


Sprint 7 - User-authored puzzles
================================

Goal: add puzzle creation last, using the final parser, solver statuses, and
accessibility contract rather than creating parallel semantics.

20. Define canonical puzzle import, export, and sharing
    Status: Partial; plain-text parsing and shared symbol fixtures already exist.

    Plan:
    - Specify a versioned interchange format containing rows, typed labels,
      optional title/author, and future-safe metadata.
    - Guarantee lossless plain-text import/export and stable board-content hashes.
    - Use one validation fixture set across browser, Python, editor, benchmarks,
      and documentation.
    - Add copy, download, and upload flows without requiring a hosted backend.
    - Reject unsupported versions and malformed metadata with actionable errors.

    Acceptance gate:
    - Round trips are lossless across browser and Python, including ragged boards
      and typed labels.

21. Add an accessible browser puzzle editor
    Status: Planned.

    Plan:
    - Support create, resize, paint/erase, typed boxes/goals, robot placement,
      undo/redo, clear, and test-play.
    - Reuse the canonical parser, game state, renderer, and movement rules.
    - Support mouse, touch, and keyboard editing with accessible tool state.
    - Preserve drafts locally and warn before destructive changes.
    - Keep editing state isolated from active solver workers and saved proofs.

    Acceptance gate:
    - Authored puzzles round-trip through the canonical format and play with the
      same semantics as built-in puzzles.

22. Add creator validation and solver-assisted publishing checks
    Status: Planned.

    Plan:
    - Report robot count, box/goal label mismatch, invalid symbols, disconnected
      floor, unreachable elements, and obvious static dead starts immediately.
    - Let authors test-play, request a solution, and attach a replay-validated
      solution certificate.
    - Distinguish proven-unsolvable, cutoff, cancelled, and failed; never describe
      a budget cutoff as impossible.
    - Provide difficulty and quality estimates only with their evidence and
      uncertainty.
    - Add generated malformed and valid-puzzle authoring fixtures.

    Acceptance gate:
    - Published puzzles pass canonical validation and, when claimed solvable,
      include a replay-valid certificate.

Sprint 7 completion gate:
Users can author, validate, solve, import, export, and share puzzles without
forking Sokomind's rules or overstating solver evidence.
