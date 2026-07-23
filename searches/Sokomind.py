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
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
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

CONFORMANCE_PATH = Path(__file__).resolve().parents[1] / "shared" / "sokomind-conformance.json"
with CONFORMANCE_PATH.open(encoding="utf-8") as conformance_file:
    BUILTIN_PUZZLES = json.load(conformance_file)["levels"]

Position = tuple[int, int]
Box = tuple[str, Position]


@dataclass(frozen=True)
class AssignmentHint:
    parent_boxes: tuple[Box, ...]
    label: str
    previous_position: Position
    position: Position


class PuzzleError(ValueError):
    """Raised when a puzzle definition is invalid."""


class SearchStatus(str, Enum):
    """Terminal meanings shared with the browser solver."""

    SOLVED = "solved"
    PROVEN_UNSOLVABLE = "proven-unsolvable"
    CUTOFF = "cutoff"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class SearchResult:
    """Structured result that remains compatible with legacy tuple unpacking."""

    status: SearchStatus
    reason: str
    path: list[tuple[str, Position]] | None
    final: State | None
    elapsed: float
    visited: int

    @property
    def cutoff(self) -> bool:
        return self.status is SearchStatus.CUTOFF

    def __iter__(self):
        yield self.path
        yield self.final
        yield self.elapsed
        yield self.visited


def _search_result(
    status: SearchStatus,
    reason: str,
    path: list[tuple[str, Position]] | None,
    final: State | None,
    elapsed: float,
    visited: int,
) -> SearchResult:
    return SearchResult(status, reason, path, final, elapsed, visited)


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
    assignment_hint: AssignmentHint | None = field(
        default=None, repr=False, compare=False, hash=False
    )

    def is_goal(self) -> bool:
        return all(pos in self.board.goals_for(label) for label, pos in self.boxes)

    @property
    def heuristic(self) -> float:
        return heuristic(self)


def _reverse_reachable(
    floor: frozenset[Position], goals: Iterable[Position]
) -> frozenset[Position]:
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
                raise PuzzleError(f"Unsupported symbol {char!r} at row {y + 1}, column {x + 1}.")
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
        raise PuzzleError(f"Puzzle must contain exactly one robot; found {len(robots)}.")
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
    goal_map.update({label: frozenset(goals) for label, goals in dedicated.items()})
    dead = {
        label: frozen_floor - _reverse_reachable(frozen_floor, goals)
        for label, goals in goal_map.items()
    }
    board = Board(
        rows,
        frozen_floor,
        frozenset(walls),
        frozenset(generic_goals),
        tuple(sorted((label, frozenset(goals)) for label, goals in dedicated.items())),
        tuple(sorted(dead.items())),
    )
    return State(robots[0], tuple(sorted(boxes)), board)


@lru_cache(maxsize=2_000)
def _reverse_push_distances(
    rows: tuple[str, ...], goal: Position
) -> tuple[tuple[Position, int], ...]:
    """Empty-board push distances from every reachable box cell to one goal."""
    floor = frozenset(
        (y, x) for y, row in enumerate(rows) for x, char in enumerate(row) if char != "O"
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


@dataclass(frozen=True)
class _Assignment:
    cost: float
    row_potential: tuple[float, ...] = ()
    column_potential: tuple[float, ...] = ()
    matching: tuple[int, ...] = ()


def _minimum_assignment(costs: Sequence[Sequence[float]]) -> _Assignment:
    """Return an exact assignment and reusable Hungarian dual state."""
    size = len(costs)
    if size == 0:
        return _Assignment(0, (0,), (0,), (0,))
    if any(len(row) != size for row in costs):
        return _Assignment(math.inf)
    blocked = 1_000_000_000.0
    finite_costs = [[blocked if math.isinf(cost) else cost for cost in row] for row in costs]
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
    if result >= blocked / 2:
        return _Assignment(math.inf)
    return _Assignment(
        result,
        tuple(row_potential),
        tuple(col_potential),
        tuple(matching),
    )


def _repair_minimum_assignment(
    previous: _Assignment,
    costs: Sequence[Sequence[float]],
    changed_row: int,
) -> _Assignment | None:
    """Repair one changed Hungarian row in O(n²), or request a full fallback."""
    size = len(costs)
    if (
        changed_row < 0
        or changed_row >= size
        or not math.isfinite(previous.cost)
        or len(previous.matching) != size + 1
        or any(len(row) != size or not any(math.isfinite(cost) for cost in row) for row in costs)
    ):
        return None
    blocked = 1_000_000_000.0
    row = changed_row + 1
    row_potential = list(previous.row_potential)
    column_potential = list(previous.column_potential)
    matching = list(previous.matching)
    freed_column = next(
        (
            column
            for column, matched_row in enumerate(matching)
            if column > 0 and matched_row == row
        ),
        -1,
    )
    if freed_column < 1:
        return None
    matching[freed_column] = 0
    row_potential[row] = min(
        (cost if math.isfinite(cost) else blocked) - column_potential[index + 1]
        for index, cost in enumerate(costs[changed_row])
    )
    matching[0] = row
    minimum = [blocked] * (size + 1)
    used = [False] * (size + 1)
    predecessor = [0] * (size + 1)
    column = 0
    while True:
        used[column] = True
        matched_row = matching[column]
        delta = blocked
        next_column = 0
        for candidate in range(1, size + 1):
            if used[candidate]:
                continue
            cost = costs[matched_row - 1][candidate - 1]
            reduced = (
                (cost if math.isfinite(cost) else blocked)
                - row_potential[matched_row]
                - column_potential[candidate]
            )
            if reduced < minimum[candidate]:
                minimum[candidate] = reduced
                predecessor[candidate] = column
            if minimum[candidate] < delta:
                delta = minimum[candidate]
                next_column = candidate
        if delta >= blocked:
            return _Assignment(math.inf)
        for candidate in range(size + 1):
            if used[candidate]:
                row_potential[matching[candidate]] += delta
                column_potential[candidate] -= delta
            else:
                minimum[candidate] -= delta
        column = next_column
        if matching[column] == 0:
            break
    while True:
        previous_column = predecessor[column]
        matching[column] = matching[previous_column]
        column = previous_column
        if column == 0:
            break
    total = 0.0
    for candidate in range(1, size + 1):
        cost = costs[matching[candidate] - 1][candidate - 1]
        if not math.isfinite(cost):
            return _Assignment(math.inf)
        total += cost
    return _Assignment(
        total,
        tuple(row_potential),
        tuple(column_potential),
        tuple(matching),
    )


def _minimum_assignment_cost(costs: Sequence[Sequence[float]]) -> float:
    """Return an exact distinct assignment cost, or infinity without a perfect matching."""
    return _minimum_assignment(costs).cost


HEURISTIC_CACHE_SIZE = 20_000
PYTHON_INCREMENTAL_ASSIGNMENT_CROSSOVER = 5
PATTERN_EXACT_STATE_LIMIT = 512
INTERACTION_PATTERN_MAX_STATES = 20_000
INTERACTION_PATTERN_MAX_FLOOR = 32
INTERACTION_PATTERN_MAX_BOXES = 3


@dataclass(frozen=True)
class _LabelAssignment:
    positions: tuple[Position, ...]
    costs: tuple[tuple[float, ...], ...]
    assignment: _Assignment


@dataclass(frozen=True)
class _AssignmentDetail:
    labels: tuple[tuple[str, _LabelAssignment], ...]
    cost: float

    def by_label(self) -> dict[str, _LabelAssignment]:
        return dict(self.labels)


_ASSIGNMENT_DETAIL_CACHE: OrderedDict[tuple[object, ...], _AssignmentDetail] = OrderedDict()
ASSIGNMENT_TELEMETRY = {
    "full_calls": 0,
    "incremental_calls": 0,
    "incremental_fallbacks": 0,
    "rows_reused": 0,
}


def _assignment_key(
    rows: tuple[str, ...],
    boxes: tuple[Box, ...],
    generic_goals: frozenset[Position],
    dedicated_goals: tuple[tuple[str, frozenset[Position]], ...],
) -> tuple[object, ...]:
    return rows, boxes, generic_goals, dedicated_goals


def _cache_assignment_detail(
    key: tuple[object, ...],
    detail: _AssignmentDetail,
) -> _AssignmentDetail:
    _ASSIGNMENT_DETAIL_CACHE[key] = detail
    _ASSIGNMENT_DETAIL_CACHE.move_to_end(key)
    while len(_ASSIGNMENT_DETAIL_CACHE) > HEURISTIC_CACHE_SIZE:
        _ASSIGNMENT_DETAIL_CACHE.popitem(last=False)
    return detail


def _full_assignment_detail(
    rows: tuple[str, ...],
    boxes: tuple[Box, ...],
    generic_goals: frozenset[Position],
    dedicated_goals: tuple[tuple[str, frozenset[Position]], ...],
) -> _AssignmentDetail:
    key = _assignment_key(rows, boxes, generic_goals, dedicated_goals)
    cached = _ASSIGNMENT_DETAIL_CACHE.get(key)
    if cached is not None:
        _ASSIGNMENT_DETAIL_CACHE.move_to_end(key)
        return cached
    goals_by_label = dict(dedicated_goals)
    goals_by_label["X"] = generic_goals
    by_label: dict[str, list[Position]] = {}
    for label, position in boxes:
        by_label.setdefault(label, []).append(position)
    details: list[tuple[str, _LabelAssignment]] = []
    total = 0.0
    for label, positions in sorted(by_label.items()):
        ordered_positions = tuple(sorted(positions))
        targets = tuple(sorted(goals_by_label.get(label, ())))
        costs = tuple(
            tuple(_push_distance(rows, position, goal) for goal in targets)
            for position in ordered_positions
        )
        assignment = _minimum_assignment(costs)
        details.append((label, _LabelAssignment(ordered_positions, costs, assignment)))
        total += assignment.cost
        ASSIGNMENT_TELEMETRY["full_calls"] += 1
    return _cache_assignment_detail(key, _AssignmentDetail(tuple(details), total))


def _incremental_assignment_detail(state: State) -> _AssignmentDetail | None:
    hint = state.assignment_hint
    if hint is None:
        return None
    board = state.board
    key = _assignment_key(
        board.rows,
        state.boxes,
        board.generic_goals,
        board.dedicated_goals,
    )
    cached = _ASSIGNMENT_DETAIL_CACHE.get(key)
    if cached is not None:
        _ASSIGNMENT_DETAIL_CACHE.move_to_end(key)
        return cached
    parent = _full_assignment_detail(
        board.rows,
        hint.parent_boxes,
        board.generic_goals,
        board.dedicated_goals,
    )
    parent_labels = parent.by_label()
    previous = parent_labels.get(hint.label)
    if previous is None or len(previous.positions) < PYTHON_INCREMENTAL_ASSIGNMENT_CROSSOVER:
        return None
    try:
        changed_row = previous.positions.index(hint.previous_position)
    except ValueError:
        ASSIGNMENT_TELEMETRY["incremental_fallbacks"] += 1
        return None
    targets = tuple(sorted(board.goals_for(hint.label)))
    positions = list(previous.positions)
    positions[changed_row] = hint.position
    costs = list(previous.costs)
    costs[changed_row] = tuple(_push_distance(board.rows, hint.position, goal) for goal in targets)
    assignment = _repair_minimum_assignment(previous.assignment, costs, changed_row)
    if assignment is None:
        ASSIGNMENT_TELEMETRY["incremental_fallbacks"] += 1
        return None
    parent_labels[hint.label] = _LabelAssignment(
        tuple(positions),
        tuple(costs),
        assignment,
    )
    detail = _AssignmentDetail(
        tuple(sorted(parent_labels.items())),
        sum(label_detail.assignment.cost for label_detail in parent_labels.values()),
    )
    ASSIGNMENT_TELEMETRY["incremental_calls"] += 1
    ASSIGNMENT_TELEMETRY["rows_reused"] += len(previous.positions) - 1
    return _cache_assignment_detail(key, detail)


@lru_cache(maxsize=HEURISTIC_CACHE_SIZE)
def _heuristic_for_layout(
    rows: tuple[str, ...],
    boxes: tuple[Box, ...],
    generic_goals: frozenset[Position],
    dedicated_goals: tuple[tuple[str, frozenset[Position]], ...],
) -> float:
    """Cache the lower bound by immutable layout data without retaining State objects."""
    assignment = _full_assignment_detail(rows, boxes, generic_goals, dedicated_goals).cost
    interaction = _interaction_pattern_bound(
        rows,
        boxes,
        generic_goals,
        dedicated_goals,
    )
    return max(assignment, interaction)


def _local_box_signature(boxes: Iterable[Box]) -> tuple[Box, ...]:
    return tuple(sorted(boxes))


@lru_cache(maxsize=256)
def _relaxed_interaction_pattern(
    rows: tuple[str, ...],
    target_boxes: tuple[Box, ...],
) -> tuple[tuple[tuple[Box, ...], int], ...]:
    floor = frozenset(
        (y, x) for y, row in enumerate(rows) for x, cell in enumerate(row) if cell != "O"
    )
    initial = _local_box_signature(target_boxes)
    distances: dict[tuple[Box, ...], int] = {initial: 0}
    queue = deque([initial])
    while queue and len(distances) < INTERACTION_PATTERN_MAX_STATES:
        current = queue.popleft()
        pushes = distances[current]
        occupied = {position for _label, position in current}
        for box_index, (label, destination) in enumerate(current):
            for dy, dx in DIRECTIONS.values():
                previous = (destination[0] - dy, destination[1] - dx)
                support = (destination[0] - 2 * dy, destination[1] - 2 * dx)
                if (
                    previous not in floor
                    or support not in floor
                    or previous in occupied
                    or support in occupied
                ):
                    continue
                predecessor = list(current)
                predecessor[box_index] = (label, previous)
                signature = _local_box_signature(predecessor)
                if signature in distances:
                    continue
                distances[signature] = pushes + 1
                queue.append(signature)
                if len(distances) >= INTERACTION_PATTERN_MAX_STATES:
                    break
            if len(distances) >= INTERACTION_PATTERN_MAX_STATES:
                break
    return tuple(distances.items())


def _interaction_pattern_bound(
    rows: tuple[str, ...],
    boxes: tuple[Box, ...],
    generic_goals: frozenset[Position],
    dedicated_goals: tuple[tuple[str, frozenset[Position]], ...],
) -> float:
    floor_size = sum(cell != "O" for row in rows for cell in row)
    if (
        len(boxes) < 2
        or len(boxes) > INTERACTION_PATTERN_MAX_BOXES
        or floor_size > INTERACTION_PATTERN_MAX_FLOOR
    ):
        return 0
    targets: list[Box] = [("X", goal) for goal in generic_goals]
    for label, goals in dedicated_goals:
        targets.extend((label, goal) for goal in goals)
    if len(targets) != len(boxes):
        return 0
    distances = dict(_relaxed_interaction_pattern(rows, _local_box_signature(targets)))
    return distances.get(_local_box_signature(boxes), 0)


def heuristic(state: State) -> float:
    """Admissible lower bound based on exact distinct box-to-goal assignment."""
    incremental = _incremental_assignment_detail(state)
    if incremental is not None:
        interaction = _interaction_pattern_bound(
            state.board.rows,
            state.boxes,
            state.board.generic_goals,
            state.board.dedicated_goals,
        )
        return max(incremental.cost, interaction)
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
            ((y - 1, x), (y, x - 1)),
            ((y - 1, x), (y, x + 1)),
            ((y + 1, x), (y, x - 1)),
            ((y + 1, x), (y, x + 1)),
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
                (origin_y, origin_x),
                (origin_y + 1, origin_x),
                (origin_y, origin_x + 1),
                (origin_y + 1, origin_x + 1),
            )
            if not all(cell in board.walls or cell in occupied for cell in cells):
                continue
            if any(
                cell in occupied and cell not in board.goals_for(occupied[cell]) for cell in cells
            ):
                return True
    return False


def creates_frozen_component_deadlock(
    boxes: Iterable[Box],
    board: Board,
    moved_box: Position,
) -> bool:
    """Detect an adjacent box group with no geometrically possible member push.

    Robot reachability is deliberately ignored: requiring both the destination
    and support cells to be free makes this a conservative hard-pruning rule.
    """
    occupied = {pos: label for label, pos in boxes}
    if moved_box not in occupied:
        return False
    component = {moved_box}
    queue = deque([moved_box])
    while queue:
        y, x = queue.popleft()
        for dy, dx in DIRECTIONS.values():
            adjacent = (y + dy, x + dx)
            if adjacent not in occupied or adjacent in component:
                continue
            component.add(adjacent)
            queue.append(adjacent)

    recursively_frozen: set[Position] = set()

    def is_blocker(position: Position) -> bool:
        return position not in board.floor or (
            position in occupied and position in recursively_frozen
        )

    changed = True
    while changed:
        changed = False
        for y, x in component - recursively_frozen:
            horizontal = is_blocker((y, x - 1)) or is_blocker((y, x + 1))
            vertical = is_blocker((y - 1, x)) or is_blocker((y + 1, x))
            if not horizontal or not vertical:
                continue
            recursively_frozen.add((y, x))
            changed = True
    if any(position not in board.goals_for(occupied[position]) for position in recursively_frozen):
        return True

    for y, x in component:
        for dy, dx in DIRECTIONS.values():
            destination = (y + dy, x + dx)
            support = (y - dy, x - dx)
            if (
                destination in board.floor
                and support in board.floor
                and destination not in occupied
                and support not in occupied
            ):
                return False
    return any(pos not in board.goals_for(occupied[pos]) for pos in component)


def creates_closed_diagonal_deadlock(
    boxes: Iterable[Box],
    board: Board,
    moved_box: Position,
) -> bool:
    """Detect a conservative wall-ended closed diagonal with no goal escape.

    This deliberately recognizes only a two-row proven subset: each open
    diagonal square is flanked by exactly one box and one wall, the boxes face
    outward toward wall endpoints, and none of the pattern squares is a goal.
    """
    occupied = {pos: label for label, pos in boxes}
    if moved_box not in occupied:
        return False
    limit = len(board.rows) + max(map(len, board.rows)) + 2
    all_goals = board.generic_goals | frozenset(
        position for _label, goals in board.dedicated_goals for position in goals
    )

    def statically_immovable(position: Position) -> bool:
        y, x = position
        return not any(
            (y + dy, x + dx) in board.floor and (y - dy, x - dx) in board.floor
            for dy, dx in DIRECTIONS.values()
        )

    def scan_half(
        start: Position,
        step: Position,
    ) -> tuple[bool, set[Position], list[int], int]:
        border_boxes: set[Position] = set()
        box_sides: list[int] = []
        y, x = start
        step_y, step_x = step
        for distance in range(limit):
            center = (y, x)
            if center in board.walls:
                return True, border_boxes, box_sides, distance
            if center in occupied and statically_immovable(center):
                border_boxes.add(center)
                return True, border_boxes, box_sides, distance
            if center not in board.floor or center in occupied or center in all_goals:
                return False, border_boxes, box_sides, distance
            row_box_side: int | None = None
            for side_offset, side in ((-1, (y, x - 1)), (1, (y, x + 1))):
                if side not in board.walls and side not in occupied:
                    return False, border_boxes, box_sides, distance
                if side in all_goals and side not in occupied:
                    return False, border_boxes, box_sides, distance
                if side in occupied:
                    if row_box_side is not None:
                        return False, border_boxes, box_sides, distance
                    row_box_side = side_offset
                    border_boxes.add(side)
            if row_box_side is None:
                return False, border_boxes, box_sides, distance
            box_sides.append(row_box_side)
            y += step_y
            x += step_x
        return False, border_boxes, box_sides, limit

    box_y, box_x = moved_box
    for center_x in (box_x - 1, box_x + 1):
        for slope in (-1, 1):
            up_closed, up_boxes, up_sides, up_rows = scan_half(
                (box_y, center_x),
                (-1, -slope),
            )
            down_closed, down_boxes, down_sides, down_rows = scan_half(
                (box_y + 1, center_x + slope),
                (1, slope),
            )
            participants = up_boxes | down_boxes
            box_sides = list(reversed(up_sides)) + down_sides
            outward_facing = (
                len(box_sides) == 2 and box_sides[0] == -slope and box_sides[1] == slope
            )
            unfinished = any(
                position not in board.goals_for(occupied[position]) for position in participants
            )
            if (
                up_closed
                and down_closed
                and up_rows + down_rows >= 2
                and unfinished
                and moved_box in participants
                and len(participants) >= 2
                and (outward_facing or creates_pattern_database_deadlock(boxes, board, moved_box))
            ):
                return True
    return False


def creates_pattern_database_deadlock(
    boxes: Iterable[Box],
    board: Board,
    moved_box: Position,
    max_states: int = PATTERN_EXACT_STATE_LIMIT,
) -> bool:
    """Exhaust a relaxed small corridor pattern, including generic boxes.

    Robot reachability is removed and a box may leave the local window, making
    the abstraction strictly more permissive than the real puzzle. A dead result
    is therefore safe; incomplete enumerations never prune.
    """
    center_y, center_x = moved_box
    local_floor = frozenset(
        position
        for position in board.floor
        if abs(position[0] - center_y) <= 4 and abs(position[1] - center_x) <= 4
    )
    if len(local_floor) > 18 or any(
        sum((position[0] + dy, position[1] + dx) in board.floor for dy, dx in DIRECTIONS.values())
        > 2
        for position in local_floor
    ):
        return False
    local_boxes = tuple(
        sorted((label, position) for label, position in boxes if position in local_floor)
    )
    if not 2 <= len(local_boxes) <= 4:
        return False
    layouts = 1
    for index in range(len(local_boxes)):
        layouts *= len(local_floor) - index
    label_counts: dict[str, int] = {}
    for label, _position in local_boxes:
        label_counts[label] = label_counts.get(label, 0) + 1
    for count in label_counts.values():
        layouts //= math.factorial(count)
    state_upper_bound = layouts * max(1, len(local_floor) - len(local_boxes))
    if state_upper_bound > PATTERN_EXACT_STATE_LIMIT:
        return False

    queue = deque([local_boxes])
    seen = {local_boxes}
    state_limit = min(max_states, PATTERN_EXACT_STATE_LIMIT, state_upper_bound + 1)
    while queue and len(seen) <= state_limit:
        current = queue.popleft()
        if all(position in board.goals_for(label) for label, position in current):
            return False
        occupied = {position for _label, position in current}
        for box_index, (label, (y, x)) in enumerate(current):
            for dy, dx in DIRECTIONS.values():
                support = (y - dy, x - dx)
                destination = (y + dy, x + dx)
                if (
                    support not in board.floor
                    or support in occupied
                    or destination not in board.floor
                    or destination in occupied
                ):
                    continue
                successor = list(current)
                if destination in local_floor:
                    successor[box_index] = (label, destination)
                else:
                    successor.pop(box_index)
                signature = tuple(sorted(successor))
                if signature in seen:
                    continue
                seen.add(signature)
                queue.append(signature)
    return not queue


def creates_dynamic_deadlock(
    boxes: Iterable[Box],
    board: Board,
    moved_box: Position,
) -> bool:
    """Combine the currently proven local multi-box deadlock rules."""
    frozen_boxes = tuple(boxes)
    return (
        creates_2x2_deadlock(frozen_boxes, board, moved_box)
        or creates_closed_diagonal_deadlock(frozen_boxes, board, moved_box)
        or creates_frozen_component_deadlock(frozen_boxes, board, moved_box)
        or creates_pattern_database_deadlock(frozen_boxes, board, moved_box)
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
        assignment_hint = None
        if label is not None:
            box_destination = (destination[0] + dy, destination[1] + dx)
            if box_destination not in state.board.floor or box_destination in occupied:
                continue
            if prune_deadlocks and is_deadlock(box_destination, state.board, box_label=label):
                continue
            moved = list(boxes)
            moved.remove((label, destination))
            moved.append((label, box_destination))
            boxes = tuple(sorted(moved))
            if prune_deadlocks and creates_dynamic_deadlock(boxes, state.board, box_destination):
                continue
            assignment_hint = AssignmentHint(
                state.boxes,
                label,
                destination,
                box_destination,
            )
        neighbors.append(
            (
                State(
                    destination,
                    boxes,
                    state.board,
                    state.cost + 1,
                    assignment_hint,
                ),
                move,
            )
        )
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
            if next_pos in parents or next_pos not in state.board.floor or next_pos in occupied:
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
            if creates_dynamic_deadlock(boxes, state.board, box_destination):
                continue
            segment = [*_reconstruct_walk(reachable, robot_support), push_move]
            neighbors.append(
                (
                    State(
                        box,
                        boxes,
                        state.board,
                        state.cost + 1,
                        AssignmentHint(state.boxes, label, box, box_destination),
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


def collapse_forced_pushes(
    state: State,
    segment: Sequence[str],
    limit: int = 32,
) -> tuple[State, list[str]]:
    """Collapse a globally forced chain while preserving its replayable walk path."""
    current = state
    moves = list(segment)
    seen = {_push_signature(current)}
    pushes = 1
    while pushes < limit and not current.is_goal():
        reachable = _reachable_parents(current)
        choices = get_push_neighbors(current, reachable)
        if len(choices) != 1:
            break
        next_state, next_segment = choices[0]
        signature = _push_signature(next_state)
        if signature in seen:
            break
        seen.add(signature)
        current = next_state
        moves.extend(next_segment)
        pushes += 1
    return current, moves


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
            return _search_result(
                SearchStatus.CANCELLED,
                "user-stop",
                None,
                None,
                time.perf_counter() - started,
                len(expanded),
            )
        if max_seconds is not None and time.perf_counter() - started >= max_seconds:
            return _search_result(
                SearchStatus.CUTOFF,
                "time-budget",
                None,
                None,
                time.perf_counter() - started,
                len(expanded),
            )
        current = pop()
        current_reachable = _reachable_parents(current) if push_macro else None
        current_key = _push_signature(current, current_reachable) if push_macro else current
        if current_key in expanded:
            continue
        expanded.add(current_key)
        if current.is_goal():
            elapsed = time.perf_counter() - started
            path = reconstruct_path(came_from, current, initial)
            return _validated_solution_result(
                initial,
                path,
                elapsed,
                len(expanded),
                expected_final=current,
            )

        if push_macro:
            children = [
                collapse_forced_pushes(neighbor, segment)
                for neighbor, segment in get_push_neighbors(current, current_reachable)
            ]
        else:
            children = [(neighbor, [move]) for neighbor, move in get_neighbors(current)]
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

    return _search_result(
        SearchStatus.PROVEN_UNSOLVABLE,
        "frontier-exhausted",
        None,
        None,
        time.perf_counter() - started,
        len(expanded),
    )


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


def push_beam_search(
    initial_state: State,
    cancel_event: Event | None = None,
    *,
    beam_width: int = 300,
    max_depth: int = 200,
    max_visited: int = 60_000,
    max_seconds: float | None = None,
):
    """Bounded best-first layers over canonical push states.

    This intentionally remains smaller than the browser portfolio, but shares
    its push-level beam terminology, deterministic budgets, and forced macros.
    """
    if beam_width < 1 or max_depth < 0 or max_visited < 1:
        raise ValueError("Beam width and state budget must be positive.")
    initial = State(initial_state.robot_pos, initial_state.boxes, initial_state.board)
    started = time.perf_counter()
    frontier = [initial]
    came_from: dict[State, tuple[State, list[str]]] = {}
    best_cost: dict[object, int] = {_push_signature(initial): 0}
    visited = 0
    order = 0

    while frontier:
        candidates: dict[object, tuple[float, int, State, State, list[str]]] = {}
        for current in frontier:
            if cancel_event is not None and cancel_event.is_set():
                return _search_result(
                    SearchStatus.CANCELLED,
                    "user-stop",
                    None,
                    None,
                    time.perf_counter() - started,
                    visited,
                )
            if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                return _search_result(
                    SearchStatus.CUTOFF,
                    "time-budget",
                    None,
                    None,
                    time.perf_counter() - started,
                    visited,
                )
            if current.is_goal():
                return _validated_solution_result(
                    initial,
                    reconstruct_path(came_from, current, initial),
                    time.perf_counter() - started,
                    visited,
                    expected_final=current,
                )
            if visited >= max_visited:
                return _search_result(
                    SearchStatus.CUTOFF,
                    "state-budget",
                    None,
                    None,
                    time.perf_counter() - started,
                    visited,
                )
            visited += 1
            reachable = _reachable_parents(current)
            for raw_state, raw_segment in get_push_neighbors(current, reachable):
                neighbor, segment = collapse_forced_pushes(raw_state, raw_segment)
                if neighbor.cost > max_depth:
                    continue
                estimate = neighbor.heuristic
                if not math.isfinite(estimate):
                    continue
                signature = _push_signature(neighbor)
                if neighbor.cost >= best_cost.get(signature, math.inf):
                    continue
                order += 1
                candidate = (
                    estimate + 0.15 * neighbor.cost,
                    order,
                    neighbor,
                    current,
                    segment,
                )
                previous = candidates.get(signature)
                if previous is None or candidate[:2] < previous[:2]:
                    candidates[signature] = candidate

        ranked = sorted(candidates.items(), key=lambda item: item[1][:2])[:beam_width]
        frontier = []
        for candidate_signature, (_score, _order, neighbor, parent, segment) in ranked:
            best_cost[candidate_signature] = neighbor.cost
            came_from[neighbor] = (parent, segment)
            frontier.append(neighbor)

    return _search_result(
        SearchStatus.CUTOFF,
        "bounded-frontier-exhausted",
        None,
        None,
        time.perf_counter() - started,
        visited,
    )


def ultimate_search(initial_state: State, cancel_event: Event | None = None):
    """Best-effort Sokoban-aware solver portfolio.

    This combines the project's current Sokoban-specific mechanics: a bounded
    push beam, forced-push macros, robot reachability canonicalization, static
    dead-square pruning, label-aware goal matching, and push-distance heuristics.
    Guided attempts are bounded before a final push-optimal A* attempt.
    """
    started = time.perf_counter()
    total_visited = 0
    attempts = (
        lambda: push_beam_search(initial_state, cancel_event, max_seconds=8),
        lambda: weighted_push_a_star_search(initial_state, cancel_event, max_seconds=20),
        lambda: push_a_star_search(initial_state, cancel_event, max_seconds=30),
    )
    for attempt in attempts:
        result = attempt()
        total_visited += result.visited
        if result.status in {SearchStatus.SOLVED, SearchStatus.CANCELLED}:
            return _search_result(
                result.status,
                result.reason,
                result.path,
                result.final,
                time.perf_counter() - started,
                total_visited,
            )
    return _search_result(
        SearchStatus.CUTOFF,
        "portfolio-budget",
        None,
        None,
        time.perf_counter() - started,
        total_visited,
    )


def portfolio_search(initial_state: State, cancel_event: Event | None = None):
    """Backward-compatible name for the ultimate Sokoban-aware search."""
    return ultimate_search(initial_state, cancel_event)


def apply_move(state: State, move: str) -> State:
    for neighbor, direction in get_neighbors(state, prune_deadlocks=False):
        if direction == move:
            return neighbor
    raise ValueError(f"Illegal move {move!r} from {state.robot_pos}.")


def _validated_solution_result(
    initial: State,
    path: Sequence[tuple[str, Position]],
    elapsed: float,
    visited: int,
    *,
    expected_final: State | None = None,
) -> SearchResult:
    """Replay a candidate independently before exposing solver success."""
    replay = State(initial.robot_pos, initial.boxes, initial.board)
    validated: list[tuple[str, Position]] = []
    try:
        for move, reported_position in path:
            replay = apply_move(replay, move)
            if replay.robot_pos != reported_position:
                return _search_result(
                    SearchStatus.FAILED,
                    "solution-position-mismatch",
                    None,
                    None,
                    elapsed,
                    visited,
                )
            validated.append((move, replay.robot_pos))
            if replay.is_goal():
                if expected_final is not None and (
                    replay.robot_pos != expected_final.robot_pos
                    or replay.boxes != expected_final.boxes
                ):
                    return _search_result(
                        SearchStatus.FAILED,
                        "solution-final-state-mismatch",
                        None,
                        None,
                        elapsed,
                        visited,
                    )
                return _search_result(
                    SearchStatus.SOLVED,
                    "solution",
                    validated,
                    expected_final or replay,
                    elapsed,
                    visited,
                )
    except ValueError:
        return _search_result(
            SearchStatus.FAILED,
            "illegal-solution-path",
            None,
            None,
            elapsed,
            visited,
        )
    return _search_result(
        SearchStatus.FAILED,
        "incomplete-solution-path",
        None,
        None,
        elapsed,
        visited,
    )


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
        raise PuzzleError(f"Could not read puzzle {file_path!s}: {exc}") from exc
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
        "push-beam": push_beam_search,
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
    source.add_argument("--puzzle", choices=BUILTIN_PUZZLES, default="ultra-tiny")
    source.add_argument("--file", type=Path, help="load a puzzle from a UTF-8 text file")
    parser.add_argument(
        "--algorithm",
        choices=(
            "astar",
            "greedy",
            "bfs",
            "dfs",
            "push-astar",
            "push-greedy",
            "push-beam",
            "weighted-push-astar",
            "ultimate",
            "portfolio",
            "fast",
        ),
        default="ultimate",
    )
    parser.add_argument("--show-steps", action="store_true")
    parser.add_argument("--output", type=Path, help="write the move list to this file")
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    try:
        puzzle = load_custom_puzzle(args.file) if args.file else BUILTIN_PUZZLES[args.puzzle]
        initial = parse_puzzle(puzzle)
        result = solve(puzzle, args.algorithm)
        path, final, elapsed, visited = result
    except (PuzzleError, ValueError) as exc:
        logging.error("%s", exc)
        return 2

    if path is None:
        print(
            f"Search ended: {result.status.value} ({result.reason}; "
            f"{visited} states, {elapsed:.3f}s)."
        )
        return 1 if result.status is not SearchStatus.FAILED else 2
    if args.output:
        try:
            args.output.write_text(
                "\n".join(f"{i}: {move} {pos}" for i, (move, pos) in enumerate(path, 1)) + "\n",
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
