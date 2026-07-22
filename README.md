# Sokomind

Sokomind is a Sokoban variant with generic and letter-matched boxes, playable
as either a Python desktop application or a static browser game. It also
includes classic DFS/BFS/Greedy/A* solvers plus **Ultimate Search**, a
Sokoban-specific portfolio using push-level search, dead-square pruning, robot
reachability canonicalization, and push-distance heuristics.

- [Desktop application setup and usage](README-desktop.md)
- [Web application documentation](docs/README.md)

Run the automated solver tests with:

```powershell
python -m unittest discover -v
node --test docs/solver-worker.test.js docs/path-validation.test.js docs/director-policy.test.js docs/keyboard-policy.test.js docs/conformance.test.js
node --test docs/pruning-differential.test.js bench/evaluator.test.js bench/generated-cases.test.js bench/benchmark.test.js bench/conformance.test.js
node bench/benchmark.js --suite smoke
node bench/benchmark.js --suite validation
```

On Windows, use `py -m unittest discover -v` if `python` is not on PATH.

The canonical level catalog and cross-runtime parsing/rule cases live in
`shared/sokomind-conformance.json`. Python loads its built-in levels from this
file; browser-embedded levels and benchmark replay rules are checked against the
same fixtures in CI.


## Proof that the huge puzzle is solvable:

![Huge Solution Proof](data/images/HugeSolutionProof.png)
