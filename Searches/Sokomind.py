"""Sokomind: a Sokoban variant with generic and dedicated boxes.

Symbols:
    O wall, R robot, X generic box, S generic goal,
    other uppercase letters are dedicated boxes, their lowercase forms are
    dedicated goals, and space is floor.
"""

from __future__ import annotations

import argparse
import heapq
import json
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

RESERVED_SYMBOLS = frozenset("ORSX")
DEDICATED_BOX_LABELS = frozenset(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - RESERVED_SYMBOLS)
DEDICATED_GOAL_LABELS = frozenset(label.lower() for label in DEDICATED_BOX_LABELS)

CONFORMANCE_PATH = (
    Path(__file__).resolve().parents[1] / "shared" / "sokomind-conformance.json"
)
with CONFORMANCE_PATH.open(encoding="utf-8") as conformance_file:
    BUILTIN_PUZZLES = json.load(conformance_file)["levels"]

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
    _dedicated_goal_map: dict[str, frozenset[Position]] = field(
        init=False, repr=False, compare=False, hash=False
    )
    _dead_square_map: dict[str, frozenset[Position]] = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_dedicated_goal_map", dict(self.dedicated_goals))
        object.__setattr__(self, "_dead_square_map", dict(self.dead_squares))

    def goals_for(self, label: str) -> frozenset[Position]:
        if label == "X":
            return self.generic_goals
        return self._dedicated_goal_map.get(label, frozenset())

    def dead_for(self, label: str) -> frozenset[Position]:
        return self._dead_square_map.get(label, frozenset())


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
    allowed = set("ORXS ") | set(DEDICATED_BOX_LABELS) | set(DEDICATED_GOAL_LABELS)
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


@lru_cache(maxsize=2_000)
def _reverse_push_distances(rows: tuple[str, ...], goal: Position) -> tuple[tuple[Position, int], ...]:
    """Empty-board push distances from every reachable box cell to one goal."""
    floor = frozenset(
        (y, x)
        for y, row in enumerate(rows)
        for x, char in enumerate(row)
        if char != "O"
    )
    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        box = queue.popleft()
        for dy, dx in DIRECTIONS.values():
            previous = (box[0] - dy, box[1] - dx)
            robot_support = (previous[0] - dy, previous[1] - dx)
            if previous in floor and robot_support in floor and previous not in distances:
                distances[previous] = distances[box] + 1
                queue.append(previous)
    return tuple(sorted(distances.items()))


def _push_distance(rows: tuple[str, ...], position: Position, goal: Position) -> float:
    return _push_distance_map(rows, goal).get(position, math.inf)


@lru_cache(maxsize=2_000)
def _push_distance_map(rows: tuple[str, ...], goal: Position) -> dict[Position, int]:
    return dict(_reverse_push_distances(rows, goal))


def _matching_cost(
    positions: tuple[Position, ...],
    goals: tuple[Position, ...],
    rows: tuple[str, ...] | None = None,
) -> float:
    """Minimum assignment cost, using Sokoban push distances when a board is supplied."""
    if len(positions) != len(goals):
        return math.inf
    size = len(positions)
    if size == 0:
        return 0
    costs = [
        [
            _push_distance(rows, pos, goal)
            if rows is not None
            else abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            for goal in goals
        ]
        for pos in positions
    ]
    return _minimum_assignment_cost(costs)


def _minimum_assignment_cost(costs: Sequence[Sequence[float]]) -> float:
    """Return an exact distinct assignment cost, or infinity without a perfect matching."""
    size = len(costs)
    if size == 0:
        return 0
    if any(len(row) != size for row in costs):
        return math.inf
    blocked = 1_000_000_000.0
    finite_costs = [
        [blocked if math.isinf(cost) else cost for cost in row]
        for row in costs
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
                    finite_costs[matched_row - 1][candidate - 1]
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
    result = -col_potential[0]
    return math.inf if result >= blocked / 2 else result


HEURISTIC_CACHE_SIZE = 20_000


@lru_cache(maxsize=HEURISTIC_CACHE_SIZE)
def _heuristic_for_layout(
    rows: tuple[str, ...],
    boxes: tuple[Box, ...],
    generic_goals: frozenset[Position],
    dedicated_goals: tuple[tuple[str, frozenset[Position]], ...],
) -> float:
    """Cache the lower bound by immutable layout data without retaining State objects."""
    goals_by_label = dict(dedicated_goals)
    goals_by_label["X"] = generic_goals
    by_label: dict[str, list[Position]] = {}
    for label, pos in boxes:
        by_label.setdefault(label, []).append(pos)
    return sum(
        _matching_cost(tuple(sorted(positions)), tuple(
            sorted(goals_by_label.get(label, ()))), rows)
        for label, positions in by_label.items()
    )


def heuristic(state: State) -> float:
    """Admissible lower bound based on exact distinct box-to-goal assignment."""
    return _heuristic_for_layout(
        state.board.rows,
        state.boxes,
        state.board.generic_goals,
        state.board.dedicated_goals,
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


def creates_2x2_deadlock(
    boxes: Iterable[Box],
    board: Board,
    moved_box: Position,
) -> bool:
    """Return True when a push creates an immovable 2x2 block.

    A 2x2 square filled entirely by walls and boxes cannot be opened by legal
    pushes. If any box in that block is not already on a matching goal, the
    branch cannot lead to a solved board.
    """
    occupied = {pos: label for label, pos in boxes}
    box_y, box_x = moved_box
    for origin_y in (box_y - 1, box_y):
        for origin_x in (box_x - 1, box_x):
            cells = (
                (origin_y, origin_x), (origin_y + 1, origin_x),
                (origin_y, origin_x + 1), (origin_y + 1, origin_x + 1),
            )
            if not all(cell in board.walls or cell in occupied for cell in cells):
                continue
            if any(
                cell in occupied and cell not in board.goals_for(occupied[cell])
                for cell in cells
            ):
                return True
    return False


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
            if prune_deadlocks and creates_2x2_deadlock(
                boxes, state.board, box_destination
            ):
                continue
        neighbors.append(
            (State(destination, boxes, state.board, state.cost + 1), move))
    return neighbors


ReachabilityParents = dict[Position, tuple[Position, str] | None]


def _reachable_parents(state: State) -> ReachabilityParents:
    """Flood fill once, storing one compact parent record per reachable cell."""
    occupied = {pos for _label, pos in state.boxes}
    parents: ReachabilityParents = {state.robot_pos: None}
    queue = deque([state.robot_pos])
    while queue:
        y, x = queue.popleft()
        for move, (dy, dx) in DIRECTIONS.items():
            next_pos = (y + dy, x + dx)
            if (
                next_pos in parents
                or next_pos not in state.board.floor
                or next_pos in occupied
            ):
                continue
            parents[next_pos] = ((y, x), move)
            queue.append(next_pos)
    return parents


def _reconstruct_walk(parents: ReachabilityParents, destination: Position) -> list[str]:
    if destination not in parents:
        raise KeyError(destination)
    path: list[str] = []
    current = destination
    while True:
        record = parents.get(current)
        if record is None:
            break
        parent, move = record
        path.append(move)
        current = parent
    path.reverse()
    return path


def get_push_neighbors(
    state: State,
    reachable: ReachabilityParents | None = None,
) -> list[tuple[State, list[str]]]:
    """Return states after one box push, including the walking path to perform it.

    Classic Sokoban solvers search primarily over pushes instead of every robot
    step. This collapses many equivalent "walk around the room" states into one
    expansion while still returning a normal step-by-step move list for the GUI.
    """
    occupied = {pos: label for label, pos in state.boxes}
    if reachable is None:
        reachable = _reachable_parents(state)
    neighbors: list[tuple[State, list[str]]] = []

    for label, box in state.boxes:
        for push_move, (dy, dx) in DIRECTIONS.items():
            robot_support = (box[0] - dy, box[1] - dx)
            box_destination = (box[0] + dy, box[1] + dx)
            if robot_support not in reachable:
                continue
            if box_destination not in state.board.floor or box_destination in occupied:
                continue
            if is_deadlock(box_destination, state.board, box_label=label):
                continue

            moved = list(state.boxes)
            moved.remove((label, box))
            moved.append((label, box_destination))
            boxes = tuple(sorted(moved))
            if creates_2x2_deadlock(boxes, state.board, box_destination):
                continue
            segment = [*_reconstruct_walk(reachable, robot_support), push_move]
            neighbors.append(
                (
                    State(
                        box,
                        boxes,
                        state.board,
                        state.cost + 1,
                    ),
                    segment,
                )
            )
    return neighbors


def _push_signature(
    state: State,
    reachable: ReachabilityParents | None = None,
) -> tuple[Position, tuple[Box, ...]]:
    if reachable is None:
        reachable = _reachable_parents(state)
    return min(reachable), state.boxes


def reconstruct_path(
    came_from: dict[State, tuple[State, list[str]]],
    end_state: State,
    initial_state: State,
):
    moves: list[str] = []
    current = end_state
    while current in came_from:
        parent, segment = came_from[current]
        moves[:0] = segment
        current = parent

    path = []
    replay = initial_state
    for move in moves:
        replay = apply_move(replay, move)
        path.append((move, replay.robot_pos))
    return path


def _search(
    initial: State,
    algorithm: str,
    cancel_event: Event | None = None,
    *,
    push_macro: bool = False,
    heuristic_weight: float = 1.0,
    max_seconds: float | None = None,
):
    # Search cost is local to this invocation. GUI states retain the player's
    # historical move count, which must not bias a new solver run.
    initial = State(initial.robot_pos, initial.boxes, initial.board)
    started = time.perf_counter()
    came_from: dict[State, tuple[State, list[str]]] = {}
    initial_key = _push_signature(initial) if push_macro else initial
    best_cost: dict[object, int] = {initial_key: 0}
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
        score = (
            initial.heuristic
            if algorithm == "greedy"
            else initial.cost + heuristic_weight * initial.heuristic
        )
        heapq.heappush(priority_queue, (score, counter, initial))

        def pop_priority() -> State:
            return heapq.heappop(priority_queue)[2]

        def push_priority(item: State) -> None:
            nonlocal counter
            counter += 1
            score = (
                item.heuristic
                if algorithm == "greedy"
                else item.cost + heuristic_weight * item.heuristic
            )
            heapq.heappush(priority_queue, (score, counter, item))

        pop = pop_priority
        push = push_priority
        has_items = lambda: bool(priority_queue)

    expanded: set[object] = set()
    while has_items():
        if cancel_event is not None and cancel_event.is_set():
            return None, None, time.perf_counter() - started, len(expanded)
        if max_seconds is not None and time.perf_counter() - started >= max_seconds:
            return None, None, time.perf_counter() - started, len(expanded)
        current = pop()
        current_reachable = _reachable_parents(current) if push_macro else None
        current_key = (
            _push_signature(current, current_reachable) if push_macro else current
        )
        if current_key in expanded:
            continue
        expanded.add(current_key)
        if current.is_goal():
            elapsed = time.perf_counter() - started
            return reconstruct_path(came_from, current, initial), current, elapsed, len(expanded)

        children = get_push_neighbors(current, current_reachable) if push_macro else [
            (neighbor, [move]) for neighbor, move in get_neighbors(current)
        ]
        if algorithm == "dfs":
            children.reverse()
        for neighbor, segment in children:
            new_cost = neighbor.cost
            neighbor_key = _push_signature(neighbor) if push_macro else neighbor
            if algorithm in {"astar", "bfs"} and new_cost >= best_cost.get(neighbor_key, math.inf):
                continue
            if neighbor_key in expanded:
                continue
            if algorithm in {"astar", "bfs"} or neighbor_key not in best_cost:
                best_cost[neighbor_key] = new_cost
                came_from[neighbor] = (current, segment)
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


def push_a_star_search(
    initial_state: State,
    cancel_event: Event | None = None,
    max_seconds: float | None = None,
):
    return _search(initial_state, "astar", cancel_event, push_macro=True, max_seconds=max_seconds)


def weighted_push_a_star_search(
    initial_state: State,
    cancel_event: Event | None = None,
    weight: float = 1.6,
    max_seconds: float | None = None,
):
    return _search(
        initial_state,
        "astar",
        cancel_event,
        push_macro=True,
        heuristic_weight=weight,
        max_seconds=max_seconds,
    )


def push_greedy_search(
    initial_state: State,
    cancel_event: Event | None = None,
    max_seconds: float | None = None,
):
    return _search(initial_state, "greedy", cancel_event, push_macro=True, max_seconds=max_seconds)


def ultimate_search(initial_state: State, cancel_event: Event | None = None):
    """Best-effort Sokoban-aware solver portfolio.

    This combines the project's current Sokoban-specific mechanics:
    push-level search, robot reachability canonicalization, static dead-square
    pruning, label-aware goal matching, and push-distance heuristics. The first
    two attempts are bounded so the GUI/browser stay responsive before falling
    back to a bounded push-optimal A* attempt.
    """
    started = time.perf_counter()
    total_visited = 0
    attempts = (
        lambda: push_greedy_search(initial_state, cancel_event, max_seconds=8),
        lambda: weighted_push_a_star_search(initial_state, cancel_event, max_seconds=20),
        lambda: push_a_star_search(initial_state, cancel_event, max_seconds=30),
    )
    for attempt in attempts:
        path, final, _elapsed, visited = attempt()
        total_visited += visited
        if path is not None or (cancel_event is not None and cancel_event.is_set()):
            return path, final, time.perf_counter() - started, total_visited
    return None, None, time.perf_counter() - started, total_visited


def portfolio_search(initial_state: State, cancel_event: Event | None = None):
    """Backward-compatible name for the ultimate Sokoban-aware search."""
    return ultimate_search(initial_state, cancel_event)


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
    searches = {
        "dfs": dfs_search,
        "bfs": bfs_search,
        "greedy": greedy_search,
        "astar": a_star_search,
        "a*": a_star_search,
        "push-astar": push_a_star_search,
        "push-a*": push_a_star_search,
        "push-greedy": push_greedy_search,
        "weighted-push-astar": weighted_push_a_star_search,
        "ultimate": ultimate_search,
        "portfolio": portfolio_search,
        "fast": ultimate_search,
    }
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
    parser.add_argument(
        "--algorithm",
        choices=(
            "astar", "greedy", "bfs", "dfs", "push-astar",
            "push-greedy", "weighted-push-astar", "ultimate", "portfolio", "fast",
        ),
        default="ultimate",
    )
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
        budget = (
            " in current search budget"
            if args.algorithm in {"fast", "portfolio", "ultimate"}
            else ""
        )
        print(f"No solution found{budget} ({visited} states, {elapsed:.3f}s).")
        return 1
    if args.output:
        try:
            args.output.write_text(
                "\n".join(f"{i}: {move} {pos}" for i, (move, pos)
                          in enumerate(path, 1)) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            logging.error("Could not write solution to %s: %s", args.output, exc)
            return 2
    print(f"Solved in {len(path)} moves ({visited} states, {elapsed:.3f}s).")
    print(" ".join(move for move, _ in path))
    if args.show_steps:
        state = initial
        print(render_puzzle(state), end="\n\n")
        for move, _ in path:
            state = apply_move(state, move)
            print(render_puzzle(state), end="\n\n")
    assert final is not None and final.is_goal()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
