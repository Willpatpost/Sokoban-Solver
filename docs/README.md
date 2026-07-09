# Sokomind Web

The web edition is a responsive, dependency-free browser game ready for GitHub
Pages. It includes five levels from Ultra Tiny through Huge, keyboard play,
undo/reset, a first-move timer, hints, and animated solving in a background Web
Worker.

The default solver is **Ultimate Bidirectional**, an experimental Sokoban-aware
search. It starts one forward Web Worker from the current board and several
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

The search budgets are intentionally bounded so the browser stays responsive on
extremely large or unsolved boards.

## How to play

Push every box onto its matching goal:

- `X` boxes go on `S` goals.
- Lettered boxes such as `A` go on the matching lowercase goal, such as `a`.
- Boxes can be pushed but cannot be pulled.

Use the arrow keys or WASD to move. Press U to undo and R to reset. The timer
starts on the first legal move.

## Local preview

From the repository root:

```powershell
python -m http.server 8000 --directory docs
```

Then open `http://localhost:8000`.

## GitHub Pages deployment

1. Push the repository to GitHub.
2. Open **Settings → Pages**.
3. Choose **GitHub Actions** as the source.
4. Push to `main` or manually run the included Pages workflow.

The `.github/workflows/pages.yml` workflow publishes this `docs/` directory.
No build command or package installation is required.
