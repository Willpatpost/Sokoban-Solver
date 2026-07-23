# Sokomind

Sokomind is a Sokoban variant with generic and letter-matched boxes, playable
as either a Python desktop application or a static browser game. It also
includes classic DFS/BFS/Greedy/A* solvers plus **Ultimate Search**, a
Sokoban-specific portfolio using push-level search, dead-square pruning, robot
reachability canonicalization, exact distinct goal assignment, and push-distance
heuristics. Its web search distinguishes temporary, conditional, and provably
packed goal placements using state-complete room, doorway, matching, and support
evidence. Guided searches keep proven boxes fixed while ordering strategic pushes
with dynamic support dependencies and cached exact searches of small rooms and
corrals. Typed doorway-flow analysis also tracks required room imports, exports,
lane direction, and staging capacity. Bounded relaxed multi-box room and
chokepoint-pair tables strengthen the admissible assignment heuristic when they
prove unavoidable box interaction cost. Guided beams reserve bounded feature-space
cells for distinct room, gate, packing, dependency, detour, and mobility states.

- [Desktop application setup and usage](README-desktop.md)
- [Web application documentation](docs/README.md)
- [Solver architecture and puzzle-independence rules](docs/ARCHITECTURE.md)

Run the automated solver tests with:

```powershell
python -m unittest discover -v
npm run test:unit
npm run check:build
npm run check:quality
node bench/performance-gate.js
node bench/verify-solution.js huge docs/optimalForHuge.txt
npm ci
npx playwright install chromium webkit
npm run test:browser
```

On Windows, use `py -m unittest discover -v` if `python` is not on PATH.
Supported development runtimes are Python 3.10 or newer and Node 20 through 24.
The Playwright suite serves `docs/` itself and exercises Chromium and WebKit.

The canonical level catalog and cross-runtime parsing/rule cases live in
`shared/sokomind-conformance.json`. Python loads its built-in levels from this
file; browser-embedded levels and benchmark replay rules are checked against the
same fixtures in CI.


## Saved Huge diagnostic solution

The saved route is a replay-valid regression artifact used to study difficult
states and pruning behavior. It is never consumed by the solver and is not a
puzzle-specific shortcut. Verify it with:

```powershell
node bench/verify-solution.js huge docs/optimalForHuge.txt
```

![Huge Solution Proof](data/images/HugeSolutionProof.png)
