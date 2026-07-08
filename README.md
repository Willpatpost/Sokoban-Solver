# Sokomind

A dependency-free Python solver for a Sokoban variant with generic and
dedicated boxes. It provides A*, greedy best-first, breadth-first, and
depth-first search.

## Puzzle format

- `O`: wall
- `R`: robot (exactly one)
- `X` / `S`: generic box / generic goal
- `A`–`Z` / `a`–`z`: dedicated box / corresponding goal
- space: floor

Each box type must have exactly as many matching goals as boxes. Short rows are
padded with walls, so malformed/ragged maps cannot create invisible floor.

## Usage

Python 3.10 or newer is required. No third-party packages are needed.

```powershell
python Searches/Sokomind.py --puzzle medium --algorithm astar
python Searches/Sokomind.py --file path/to/puzzle.txt --show-steps
python Searches/Sokomind.py --help
```

Use `--output solution.txt` to export moves. The legacy `astar.py`, `bfs.py`,
`dfs.py`, and `greedy.py` files remain as small compatibility entry points.

## Desktop application

Launch the playable Tkinter application with:

```powershell
python Searches/gui.py
```

Use the arrow keys or WASD to move. The application includes undo and reset,
built-in and custom puzzles, animated solver playback, and hints from the
current position. Solver work runs in a background thread so the window remains
responsive.

## Web application

The dependency-free browser version lives in `docs/` and is ready for GitHub
Pages. It includes keyboard play, responsive level cards, undo/reset, hints,
animated solving, and the level-complete flow.

Enable **GitHub Pages → GitHub Actions** in the repository settings, then push
to `main`. The included workflow deploys the site automatically. To preview it
locally:

```powershell
python -m http.server 8000 --directory docs
```

## Tests

```powershell
python -m unittest discover -v
```

The solver validates input, uses immutable search states, detects all static
dead squares by reverse box reachability, and uses an admissible minimum
box-to-goal assignment heuristic for A*.
