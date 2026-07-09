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
```
