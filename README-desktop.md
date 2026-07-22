# Sokomind Desktop

The desktop edition is a Tkinter application for Windows, macOS, and Linux. It
includes five built-in levels, custom puzzle loading, keyboard play, a
first-move timer, undo/reset, hints, and animated solver playback.
The move-history panel records every direction and marks box pushes; use Copy to
export a replayable numbered list.

## Requirements

- Python 3.10 or newer
- Tkinter

Tkinter ships with standard Python installers on Windows and macOS. On Debian
or Ubuntu Linux, install it with:

```bash
sudo apt update
sudo apt install python3-tk
```

There are no PyPI dependencies.

## Setup

Clone the repository and enter its directory:

```bash
git clone https://github.com/Willpatpost/Sokomind.git
cd Sokomind
```

Creating a virtual environment is optional because the project has no external
packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with `source .venv/bin/activate`.

## Run the application

```powershell
python Searches/gui.py
```

Controls:

- Arrow keys or WASD: move
- Backspace or U: undo
- R: reset
- Hint: show the solver's next suggested move
- Solve: animate a complete solution
- Stop: cancel the current search or animation
- Home: return to the welcome screen

The timer begins only after the first legal move.

## Command-line solver

```powershell
python Searches/Sokomind.py --puzzle medium --algorithm ultimate
python Searches/Sokomind.py --puzzle large --algorithm push-greedy
python Searches/Sokomind.py --puzzle medium --algorithm astar
python Searches/Sokomind.py --file path/to/puzzle.txt --show-steps
python Searches/Sokomind.py --help
```

Recommended algorithms:

- `ultimate`: recommended mode; combines Sokoban-specific mechanics including
  push-level search, dead-square pruning, robot-reachability canonicalization,
  exact distinct label-aware goal matching, and push-distance heuristics.
- `fast` / `portfolio`: older aliases for the same recommended portfolio.
- `push-greedy`: usually the quickest way to find a playable solution on big
  levels, though it may not be shortest.
- `push-astar`: searches at the box-push level instead of every walking step.
- `astar`, `greedy`, `bfs`, `dfs`: classic step-by-step searches, mostly useful
  for comparison and small puzzles.

`ultimate` is bounded so the app stays responsive. If it cannot solve a very
large board, try `push-greedy` directly or expect to add stronger
Sokoban-specific deadlock/pattern-database logic for that level.

The local solver uses exact polynomial assignment for every box count, including
Hall-failure detection when no distinct box-to-goal matching exists. Its bounded
layout-only heuristic cache avoids retaining complete search states, and
push-level reachability stores compact parent directions so walking paths are
reconstructed only for legal retained pushes.

Puzzle symbols are `O` wall, `R` robot, `X` generic box, `S` generic goal,
and spaces for floor. Other uppercase letters are dedicated boxes and their
lowercase forms are the matching goals; `O`, `R`, `S`, and `X` are reserved
and cannot be used as dedicated labels.

## Tests

```powershell
python -m unittest discover -v
```
