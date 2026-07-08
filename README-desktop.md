# Sokomind Desktop

The desktop edition is a Tkinter application for Windows, macOS, and Linux. It
includes five built-in levels, custom puzzle loading, keyboard play, a
first-move timer, undo/reset, hints, and animated solver playback.

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
git clone https://github.com/YOUR-USERNAME/Sokoban-Solver.git
cd Sokoban-Solver
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
python Searches/Sokomind.py --puzzle medium --algorithm astar
python Searches/Sokomind.py --file path/to/puzzle.txt --show-steps
python Searches/Sokomind.py --help
```

Puzzle symbols are `O` wall, `R` robot, `X` generic box, `S` generic goal,
uppercase letters for dedicated boxes, lowercase letters for their goals, and
spaces for floor.

## Tests

```powershell
python -m unittest discover -v
```
