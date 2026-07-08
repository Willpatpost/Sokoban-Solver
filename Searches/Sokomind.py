"""Sokomind: a Sokoban variant with generic and dedicated boxes.

Symbols:
    O wall, R robot, X generic box, S generic goal,
    A-Z dedicated boxes, a-z their dedicated goals, space floor.
"""

from __future__ import annotations

import argparse
import heapq
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Event
from typing import Callable, Iterable, Sequence

DIRECTIONS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Right": (0, 1),
    "Left": (0, -1),
}

BUILTIN_PUZZLES = {
    "ultra-tiny": ["OOOOO", "O R O", "O A O", "O a O", "OOOOO"],
    "tiny": ["OOOOOO", "O R  O", "O XO O", "OO A O", "OSa  O", "OOOOOO"],
    "medium": [
        "OOOOOOO", "Oa   bO", "O AXB O", "O XRX O",
        "OSCXDSO", "OcS SdO", "OOOOOOO",
    ],
    "large": [
        "OOOOOOOOOO", "OOOOOOOSSO", "OOOOO  abO", "OOOOO XSSO",
        "OOOOOO  OO", "OR     OOO", "OO A X X O", "OO BXO O O",
        "OO   O   O", "OOOOOOOOOO",
    ],
    "huge": [
        "OOOOOOOOOOOOOOO", "OaSS   S   SSbO", "OSCS       SDSO",
        "OX X       X XO", "O             O", "OOOO   X   OOOO",
        "O      O      O", "O G hOOOOOH g O", "O      O      O",
        "O             O", "O     X X     O", "OOOOOOOROOOOOOO",
        "O B X X X X A O", "O Sc       dS O", "OOOOOOOOOOOOOOO",
    ],
}

Position = tuple[int, int]
Box = tuple[str, Position]


class PuzzleError(ValueError):
    """Raised when a puzzle definition is invalid."""


@dataclass(frozen=True)
class Board:
    rows: tuple[str, ...]
    floor: frozenset[Position]
    walls: frozenset[Position]
    generic_goals: frozenset[Position]
    dedicated_goals: tuple[tuple[str, frozenset[Position]], ...]
    dead_squares: tuple[tuple[str, frozenset[Position]], ...]

    def goals_for(self, label: str) -> frozenset[Position]:
        if label == "X":
            return self.generic_goals
        return dict(self.dedicated_goals).get(label, frozenset())

    def dead_for(self, label: str) -> frozenset[Position]:
        return dict(self.dead_squares).get(label, frozenset())


@dataclass(frozen=True)
class State:
    robot_pos: Position
    boxes: tuple[Box, ...]
    board: Board = field(repr=False)
    cost: int = field(default=0, compare=False, hash=False)

    def is_goal(self) -> bool:
        return all(pos in self.board.goals_for(label) for label, pos in self.boxes)

    @property
    def heuristic(self) -> float:
        return heuristic(self)

    @property
    def priority(self) -> float:
        return self.cost + self.heuristic

    def calculate_heuristic(self, algorithm: str = "astar") -> float:
        value = self.heuristic if algorithm.lower(
        ) in {"astar", "a*", "greedy"} else 0
        return value if algorithm.lower() == "greedy" else self.cost + value


def _reverse_reachable(floor: frozenset[Position], goals: Iterable[Position]) -> frozenset[Position]:
    """Cells from which a box can reach a goal on an otherwise empty board."""
    reachable = set(goals)
    queue = deque(reachable)
    while queue:
        box = queue.popleft()
        for dy, dx in DIRECTIONS.values():
            previous = (box[0] - dy, box[1] - dx)
            robot_support = (previous[0] - dy, previous[1] - dx)
            if previous in floor and robot_support in floor and previous not in reachable:
                reachable.add(previous)
                queue.append(previous)
    return frozenset(reachable)


def parse_puzzle(puzzle: Sequence[str]) -> State:
    if not puzzle:
        raise PuzzleError("Puzzle is empty.")
    if any(not isinstance(row, str) for row in puzzle):
        raise PuzzleError("Every puzzle row must be a string.")

    width = max(map(len, puzzle))
    rows = tuple(row.ljust(width, "O") for row in puzzle)
    allowed = set("ORXS ") | set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | set(
        "abcdefghijklmnopqrstuvwxyz")
    robots: list[Position] = []
    boxes: list[Box] = []
    walls: set[Position] = set()
    floor: set[Position] = set()
    generic_goals: set[Position] = set()
    dedicated: dict[str, set[Position]] = {}

    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            if char not in allowed:
                raise PuzzleError(
                    f"Unsupported symbol {char!r} at row {y + 1}, column {x + 1}.")
            pos = (y, x)
            if char == "O":
                walls.add(pos)
                continue
            floor.add(pos)
            if char == "R":
                robots.append(pos)
            elif char == "X":
                boxes.append(("X", pos))
            elif char == "S":
                generic_goals.add(pos)
            elif "A" <= char <= "Z":
                boxes.append((char, pos))
            elif "a" <= char <= "z":
                dedicated.setdefault(char.upper(), set()).add(pos)

    if len(robots) != 1:
        raise PuzzleError(
            f"Puzzle must contain exactly one robot; found {len(robots)}.")
    if len({pos for _, pos in boxes}) != len(boxes):
        raise PuzzleError("Two boxes occupy the same cell.")

    labels = set(dedicated) | {label for label, _ in boxes if label != "X"}
    for label in sorted(labels):
        box_count = sum(box_label == label for box_label, _ in boxes)
        goal_count = len(dedicated.get(label, ()))
        if box_count != goal_count:
            raise PuzzleError(
                f"Dedicated box {label!r} has {box_count} box(es) but {goal_count} goal(s)."
            )
    generic_count = sum(label == "X" for label, _ in boxes)
    if generic_count != len(generic_goals):
        raise PuzzleError(
            f"Generic boxes/goals mismatch: {generic_count} box(es), {len(generic_goals)} goal(s)."
        )

    frozen_floor = frozenset(floor)
    goal_map = {"X": frozenset(generic_goals)}
    goal_map.update({label: frozenset(goals)
                    for label, goals in dedicated.items()})
    dead = {
        label: frozen_floor - _reverse_reachable(frozen_floor, goals)
        for label, goals in goal_map.items()
    }
    board = Board(
        rows, frozen_floor, frozenset(walls), frozenset(generic_goals),
        tuple(sorted((label, frozenset(goals))
              for label, goals in dedicated.items())),
        tuple(sorted(dead.items())),
    )
    return State(robots[0], tuple(sorted(boxes)), board)


def _matching_cost(positions: tuple[Position, ...], goals: tuple[Position, ...]) -> float:
    """Minimum assignment cost via the O(n^3) Hungarian algorithm."""
    if len(positions) != len(goals):
        return math.inf
    size = len(positions)
    if size == 0:
        return 0
    costs = [
        [abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) for goal in goals]
        for pos in positions
    ]
    row_potential: list[float] = [0.0] * (size + 1)
    col_potential: list[float] = [0.0] * (size + 1)
    matching: list[int] = [0] * (size + 1)
    predecessor: list[int] = [0] * (size + 1)
    for row in range(1, size + 1):
        matching[0] = row
        min_value = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        column = 0
        while True:
            used[column] = True
            matched_row = matching[column]
            delta = math.inf
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[matched_row - 1][candidate - 1]
                    - row_potential[matched_row]
                    - col_potential[candidate]
                )
                if reduced < min_value[candidate]:
                    min_value[candidate] = reduced
                    predecessor[candidate] = column
                if min_value[candidate] < delta:
                    delta = min_value[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    row_potential[matching[candidate]] += delta
                    col_potential[candidate] -= delta
                else:
                    min_value[candidate] -= delta
            column = next_column
            if matching[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matching[column] = matching[previous]
            column = previous
            if column == 0:
                break
    return -col_potential[0]


@lru_cache(maxsize=200_000)
def heuristic(state: State) -> float:
    """Admissible lower bound based on optimal box-to-goal assignment."""
    by_label: dict[str, list[Position]] = {}
    for label, pos in state.boxes:
        by_label.setdefault(label, []).append(pos)
    return sum(
        _matching_cost(tuple(sorted(positions)), tuple(
            sorted(state.board.goals_for(label))))
        for label, positions in by_label.items()
    )


def is_deadlock(box_pos: Position, walls_or_board, goals=None, box_label: str = "X") -> bool:
    """Compatibility helper; the Board form performs complete static-dead-square checks."""
    if isinstance(walls_or_board, Board):
        return box_pos in walls_or_board.dead_for(box_label)
    walls = set(walls_or_board)
    goal_positions = set((goals or {}).get(box_label, ()))
    if box_pos in goal_positions:
        return False
    y, x = box_pos
    return any(
        a in walls and b in walls
        for a, b in [
            ((y - 1, x), (y, x - 1)), ((y - 1, x), (y, x + 1)),
            ((y + 1, x), (y, x - 1)), ((y + 1, x), (y, x + 1)),
        ]
    )


def get_neighbors(
    state: State,
    algorithm: str = "astar",
    *,
    prune_deadlocks: bool = True,
) -> list[tuple[State, str]]:
    """Return legal moves, optionally omitting pushes that can never reach a goal.

    Search algorithms enable deadlock pruning. Interactive play disables it so
    the player remains free to make any mechanically legal move.
    """
    del algorithm  # Kept for backward compatibility.
    occupied = {pos: label for label, pos in state.boxes}
    neighbors = []
    for move, (dy, dx) in DIRECTIONS.items():
        destination = (state.robot_pos[0] + dy, state.robot_pos[1] + dx)
        if destination not in state.board.floor:
            continue
        label = occupied.get(destination)
        boxes = state.boxes
        if label is not None:
            box_destination = (destination[0] + dy, destination[1] + dx)
            if box_destination not in state.board.floor or box_destination in occupied:
                continue
            if prune_deadlocks and is_deadlock(
                box_destination, state.board, box_label=label
            ):
                continue
            moved = list(boxes)
            moved.remove((label, destination))
            moved.append((label, box_destination))
            boxes = tuple(sorted(moved))
        neighbors.append(
            (State(destination, boxes, state.board, state.cost + 1), move))
    return neighbors


def reconstruct_path(came_from: dict[State, tuple[State, str]], end_state: State):
    path = []
    current = end_state
    while current in came_from:
        parent, move = came_from[current]
        path.append((move, current.robot_pos))
        current = parent
    return list(reversed(path))


def _search(initial: State, algorithm: str, cancel_event: Event | None = None):
    started = time.perf_counter()
    came_from: dict[State, tuple[State, str]] = {}
    best_cost = {initial: 0}
    counter = 0
    pop: Callable[[], State]
    push: Callable[[State], None]
    has_items: Callable[[], bool]

    if algorithm == "bfs":
        queue: deque[State] = deque([initial])
        pop = lambda: queue.popleft()
        push = lambda item: queue.append(item)
        has_items = lambda: bool(queue)
    elif algorithm == "dfs":
        stack: list[State] = [initial]
        pop = lambda: stack.pop()
        push = lambda item: stack.append(item)
        has_items = lambda: bool(stack)
    else:
        priority_queue: list[tuple[float, int, State]] = []
        score = initial.heuristic if algorithm == "greedy" else initial.priority
        heapq.heappush(priority_queue, (score, counter, initial))

        def pop_priority() -> State:
            return heapq.heappop(priority_queue)[2]

        def push_priority(item: State) -> None:
            nonlocal counter
            counter += 1
            score = item.heuristic if algorithm == "greedy" else item.priority
            heapq.heappush(priority_queue, (score, counter, item))

        pop = pop_priority
        push = push_priority
        has_items = lambda: bool(priority_queue)

    expanded: set[State] = set()
    while has_items():
        if cancel_event is not None and cancel_event.is_set():
            return None, None, time.perf_counter() - started, len(expanded)
        current = pop()
        if current in expanded:
            continue
        expanded.add(current)
        if current.is_goal():
            elapsed = time.perf_counter() - started
            return reconstruct_path(came_from, current), current, elapsed, len(expanded)

        children = get_neighbors(current)
        if algorithm == "dfs":
            children.reverse()
        for neighbor, move in children:
            new_cost = current.cost + 1
            if algorithm in {"astar", "bfs"} and new_cost >= best_cost.get(neighbor, math.inf):
                continue
            if neighbor in expanded:
                continue
            if algorithm in {"astar", "bfs"} or neighbor not in came_from:
                best_cost[neighbor] = new_cost
                came_from[neighbor] = (current, move)
                push(neighbor)

    return None, None, time.perf_counter() - started, len(expanded)


def dfs_search(initial_state: State, cancel_event: Event | None = None):
    return _search(initial_state, "dfs", cancel_event)


def bfs_search(initial_state: State, cancel_event: Event | None = None):
    return _search(initial_state, "bfs", cancel_event)


def greedy_search(initial_state: State, cancel_event: Event | None = None):
    return _search(initial_state, "greedy", cancel_event)


def a_star_search(initial_state: State, cancel_event: Event | None = None):
    return _search(initial_state, "astar", cancel_event)


def apply_move(state: State, move: str) -> State:
    for neighbor, direction in get_neighbors(state, prune_deadlocks=False):
        if direction == move:
            return neighbor
    raise ValueError(f"Illegal move {move!r} from {state.robot_pos}.")


def render_puzzle(state: State) -> str:
    grid = [list(row) for row in state.board.rows]
    for y, row in enumerate(grid):
        for x, char in enumerate(row):
            if char not in {"O", "S"} and not ("a" <= char <= "z"):
                grid[y][x] = " "
    for label, (y, x) in state.boxes:
        grid[y][x] = label
    y, x = state.robot_pos
    grid[y][x] = "R"
    return "\n".join("".join(row) for row in grid)


def print_puzzle(state: State) -> None:
    print(render_puzzle(state))


def load_custom_puzzle(file_path: str | Path) -> list[str]:
    try:
        with Path(file_path).open(encoding="utf-8") as handle:
            rows = [line.rstrip("\r\n") for line in handle]
    except OSError as exc:
        raise PuzzleError(
            f"Could not read puzzle {file_path!s}: {exc}") from exc
    while rows and not rows[-1]:
        rows.pop()
    return rows


def solve(puzzle: Sequence[str], algorithm: str = "astar"):
    searches = {"dfs": dfs_search, "bfs": bfs_search, "greedy": greedy_search,
                "astar": a_star_search, "a*": a_star_search}
    key = algorithm.lower()
    if key not in searches:
        raise ValueError(f"Unknown algorithm {algorithm!r}.")
    return searches[key](parse_puzzle(puzzle))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--puzzle", choices=BUILTIN_PUZZLES,
                        default="ultra-tiny")
    source.add_argument("--file", type=Path,
                        help="load a puzzle from a UTF-8 text file")
    parser.add_argument("--algorithm", choices=("astar",
                        "greedy", "bfs", "dfs"), default="astar")
    parser.add_argument("--show-steps", action="store_true")
    parser.add_argument("--output", type=Path,
                        help="write the move list to this file")
    parser.add_argument("--log-level", choices=("DEBUG",
                        "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s: %(message)s")
    try:
        puzzle = load_custom_puzzle(
            args.file) if args.file else BUILTIN_PUZZLES[args.puzzle]
        initial = parse_puzzle(puzzle)
        path, final, elapsed, visited = solve(puzzle, args.algorithm)
    except (PuzzleError, ValueError) as exc:
        logging.error("%s", exc)
        return 2

    if path is None:
        print(f"No solution found ({visited} states, {elapsed:.3f}s).")
        return 1
    print(f"Solved in {len(path)} moves ({visited} states, {elapsed:.3f}s).")
    print(" ".join(move for move, _ in path))
    if args.show_steps:
        state = initial
        print(render_puzzle(state), end="\n\n")
        for move, _ in path:
            state = apply_move(state, move)
            print(render_puzzle(state), end="\n\n")
    if args.output:
        args.output.write_text(
            "\n".join(f"{i}: {move} {pos}" for i, (move, pos)
                      in enumerate(path, 1)) + "\n",
            encoding="utf-8",
        )
    assert final is not None and final.is_goal()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
