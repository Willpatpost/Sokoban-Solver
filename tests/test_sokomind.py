import io
import json
import tempfile
import unittest
from collections import deque
from contextlib import redirect_stdout
from itertools import combinations
from pathlib import Path
from threading import Event

from Searches.Sokomind import (
    BUILTIN_PUZZLES,
    CONFORMANCE_PATH,
    HEURISTIC_CACHE_SIZE,
    PuzzleError,
    SearchStatus,
    _heuristic_for_layout,
    _minimum_assignment_cost,
    _reachable_parents,
    _reconstruct_walk,
    apply_move,
    a_star_search,
    collapse_forced_pushes,
    creates_closed_diagonal_deadlock,
    creates_frozen_component_deadlock,
    get_push_neighbors,
    get_neighbors,
    main,
    parse_puzzle,
    push_beam_search,
    push_a_star_search,
    render_puzzle,
    solve,
)


class SokomindTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conformance = json.loads(CONFORMANCE_PATH.read_text(encoding="utf-8"))

    @staticmethod
    def _reference_min_pushes(initial):
        """Independent 0-1 step-state search with no production pruning."""
        distances = {initial: 0}
        queue = deque([initial])
        while queue:
            current = queue.popleft()
            cost = distances[current]
            if current.is_goal():
                return cost
            for child, _move in get_neighbors(current, prune_deadlocks=False):
                pushed = child.boxes != current.boxes
                next_cost = cost + int(pushed)
                if next_cost >= distances.get(child, float("inf")):
                    continue
                distances[child] = next_cost
                if pushed:
                    queue.append(child)
                else:
                    queue.appendleft(child)
        return None

    def test_shared_valid_conformance_cases(self):
        for case in self.conformance["validCases"]:
            with self.subTest(case=case["id"]):
                expected = case["expected"]
                state = parse_puzzle(case["rows"])
                goals = [(position, "X") for position in state.board.generic_goals]
                for label, positions in state.board.dedicated_goals:
                    goals.extend((position, label) for position in positions)
                boxes = sorted((f"{y},{x}", label) for label, (y, x) in state.boxes)
                encoded_goals = sorted((f"{y},{x}", label) for (y, x), label in goals)

                self.assertEqual(expected["width"], len(state.board.rows[0]))
                self.assertEqual(expected["height"], len(state.board.rows))
                self.assertEqual(expected["floorCount"], len(state.board.floor))
                self.assertEqual(expected["robot"], ",".join(map(str, state.robot_pos)))
                self.assertEqual(expected["boxes"], [list(item) for item in boxes])
                self.assertEqual(expected["goals"], [list(item) for item in encoded_goals])
                self.assertEqual(
                    expected["legalMoves"],
                    sorted(move for _next_state, move in get_neighbors(state)),
                )
                self.assertEqual(
                    expected["mechanicalMoves"],
                    sorted(
                        move for _next_state, move in get_neighbors(state, prune_deadlocks=False)
                    ),
                )
                self.assertEqual(expected["solved"], state.is_goal())
                if "missingWall" in expected:
                    missing = tuple(map(int, expected["missingWall"].split(",")))
                    self.assertNotIn(missing, state.board.floor)

    def test_shared_invalid_conformance_cases(self):
        patterns = {
            "symbol": "Unsupported symbol",
            "robot-count": "exactly one robot",
            "box-goal-count": "mismatch|box\\(es\\)",
        }
        for case in self.conformance["invalidCases"]:
            with (
                self.subTest(case=case["id"]),
                self.assertRaisesRegex(PuzzleError, patterns[case["errorKind"]]),
            ):
                parse_puzzle(case["rows"])

    def test_every_builtin_puzzle_is_valid(self):
        self.assertEqual(
            {"ultra-tiny", "tiny", "medium", "large", "huge"},
            set(BUILTIN_PUZZLES),
        )
        for name, puzzle in BUILTIN_PUZZLES.items():
            with self.subTest(name=name):
                parse_puzzle(puzzle)

    def test_all_searches_solve_simple_puzzle(self):
        puzzle = ["OOOOO", "O R O", "O A O", "O a O", "OOOOO"]
        for algorithm in (
            "astar",
            "greedy",
            "bfs",
            "dfs",
            "push-astar",
            "push-beam",
            "fast",
            "ultimate",
        ):
            with self.subTest(algorithm=algorithm):
                path, final, _, _ = solve(puzzle, algorithm)
                self.assertEqual(["Down"], [move for move, _ in path])
                self.assertTrue(final.is_goal())

    def test_generic_boxes_are_matched_to_distinct_goals(self):
        puzzle = [
            "OOOOOOO",
            "OS   SO",
            "O X X O",
            "O  R  O",
            "OOOOOOO",
        ]
        state = parse_puzzle(puzzle)
        self.assertEqual(4, state.heuristic)

    def test_large_assignment_is_exact_instead_of_reusing_goal_minima(self):
        costs = [[float("inf")] * 9 for _ in range(9)]
        costs[0][0] = 0
        for index in range(1, 9):
            costs[index][0] = 0
            costs[index][index] = index
        self.assertEqual(36, _minimum_assignment_cost(costs))

    def test_large_assignment_detects_hall_failure(self):
        costs = [[float("inf")] * 9 for _ in range(9)]
        costs[0][0] = 0
        costs[1][0] = 0
        for index in range(2, 9):
            costs[index][index] = 0
        self.assertEqual(float("inf"), _minimum_assignment_cost(costs))

    def test_heuristic_cache_is_bounded_and_keyed_by_layout(self):
        state = parse_puzzle(["OOOOO", "O R O", "O A O", "O a O", "OOOOO"])
        _heuristic_for_layout.cache_clear()
        self.assertEqual(1, state.heuristic)
        self.assertEqual(1, state.heuristic)
        info = _heuristic_for_layout.cache_info()
        self.assertEqual(HEURISTIC_CACHE_SIZE, info.maxsize)
        self.assertEqual(1, info.hits)
        self.assertEqual(1, info.currsize)

    def test_reachability_reconstructs_paths_only_when_requested(self):
        state = parse_puzzle(
            [
                "OOOOOOO",
                "O  R  O",
                "O     O",
                "O  X SO",
                "OOOOOOO",
            ]
        )
        parents = _reachable_parents(state)
        self.assertEqual(["Down", "Left", "Down"], _reconstruct_walk(parents, (3, 2)))
        self.assertIsNone(parents[state.robot_pos])
        self.assertTrue(
            all(record is None or isinstance(record, tuple) for record in parents.values())
        )

    def test_ragged_missing_cells_are_walls(self):
        state = parse_puzzle(["OOOOO", "OR  O", "OOOO", "OaA O", "OOOOO"])
        self.assertNotIn((2, 4), state.board.floor)

    def test_rejects_invalid_counts_and_symbols(self):
        bad_puzzles = [
            ["OOO", "ORO", "OXO", "OOO"],
            ["OOO", "ORO", "O?O", "OOO"],
            ["OOO", "O O", "OOO"],
            ["OOOO", "ORRO", "OOOO"],
        ]
        for puzzle in bad_puzzles:
            with self.subTest(puzzle=puzzle), self.assertRaises(PuzzleError):
                parse_puzzle(puzzle)

    def test_rejects_lowercase_goals_for_reserved_symbols(self):
        for reserved_goal in "orsx":
            puzzle = ["OOOOO", f"OR {reserved_goal}O", "OOOOO"]
            with (
                self.subTest(goal=reserved_goal),
                self.assertRaisesRegex(PuzzleError, "Unsupported symbol"),
            ):
                parse_puzzle(puzzle)

    def test_static_dead_square_push_is_pruned(self):
        state = parse_puzzle(
            [
                "OOOOOO",
                "O    O",
                "O RX O",
                "O  S O",
                "OOOOOO",
            ]
        )
        # Pushing right strands the box against the right wall, away from its goal.
        moves = [move for _, move in get_neighbors(state)]
        self.assertNotIn("Right", moves)

    def test_player_can_make_legal_push_into_dead_square(self):
        state = parse_puzzle(
            [
                "OOOOOO",
                "O    O",
                "O RX O",
                "O  S O",
                "OOOOOO",
            ]
        )
        pushed = apply_move(state, "Right")
        self.assertIn(("X", (2, 4)), pushed.boxes)

    def test_static_2x2_box_deadlock_push_is_pruned(self):
        state = parse_puzzle(
            [
                "OOOOOO",
                "O    O",
                "O  XXO",
                "ORX OO",
                "O SSSO",
                "OOOOOO",
            ]
        )
        moves = [move for _, move in get_neighbors(state)]
        self.assertNotIn("Right", moves)

    def test_frozen_components_reject_only_immovable_unfinished_groups(self):
        frozen = parse_puzzle(["OOOOOOOOO", "ORXXXSSSO", "OOOOOOOOO"])
        self.assertTrue(
            creates_frozen_component_deadlock(
                frozen.boxes,
                frozen.board,
                (1, 3),
            )
        )

        movable = parse_puzzle(
            [
                "OOOOOOOOO",
                "O       O",
                "O XXXSSSO",
                "OR      O",
                "OOOOOOOOO",
            ]
        )
        self.assertFalse(
            creates_frozen_component_deadlock(
                movable.boxes,
                movable.board,
                (2, 3),
            )
        )

    def test_closed_diagonal_requires_wall_ends_multiple_boxes_and_no_goal_escape(self):
        state = parse_puzzle(
            [
                "OOOOOOOO",
                "O O    O",
                "O X O  O",
                "O  O X O",
                "O    O O",
                "O RSS  O",
                "OOOOOOOO",
            ]
        )
        self.assertTrue(
            creates_closed_diagonal_deadlock(
                state.boxes,
                state.board,
                (2, 2),
            )
        )

        escaped = parse_puzzle(
            [
                "OOOOOOOO",
                "O O    O",
                "O XS O O",
                "O  O X O",
                "O    O O",
                "O R S  O",
                "OOOOOOOO",
            ]
        )
        self.assertFalse(
            creates_closed_diagonal_deadlock(
                escaped.boxes,
                escaped.board,
                (2, 2),
            )
        )

    def test_frozen_component_pruning_preserves_exhaustively_solvable_tiny_pushes(self):
        interior = [(y, x) for y in (1, 2) for x in (1, 2, 3)]
        solvable_layouts = 0
        checked_pushes = 0
        checked_adjacent_groups = 0
        for goal_indexes in combinations(range(len(interior)), 2):
            remaining = [index for index in range(len(interior)) if index not in goal_indexes]
            for box_indexes in combinations(remaining, 2):
                robot_indexes = [index for index in remaining if index not in box_indexes]
                for robot_index in robot_indexes:
                    rows = [list("OOOOO") for _ in range(4)]
                    for y, x in interior:
                        rows[y][x] = " "
                    for index in goal_indexes:
                        y, x = interior[index]
                        rows[y][x] = "S"
                    for index in box_indexes:
                        y, x = interior[index]
                        rows[y][x] = "X"
                    robot_y, robot_x = interior[robot_index]
                    rows[robot_y][robot_x] = "R"
                    initial = parse_puzzle(["".join(row) for row in rows])

                    states = {initial}
                    edges: dict[object, list[object]] = {}
                    reverse: dict[object, set[object]] = {}
                    queue = deque([initial])
                    while queue:
                        parent = queue.popleft()
                        children = [
                            child
                            for child, _move in get_neighbors(
                                parent,
                                prune_deadlocks=False,
                            )
                        ]
                        edges[parent] = children
                        for child in children:
                            reverse.setdefault(child, set()).add(parent)
                            if child in states:
                                continue
                            states.add(child)
                            queue.append(child)

                    solvable = {state for state in states if state.is_goal()}
                    solved_queue = deque(solvable)
                    while solved_queue:
                        child = solved_queue.popleft()
                        for parent in reverse.get(child, set()):
                            if parent in solvable:
                                continue
                            solvable.add(parent)
                            solved_queue.append(parent)
                    if not solvable:
                        continue
                    solvable_layouts += 1

                    for parent in solvable:
                        retained = {
                            (child.robot_pos, child.boxes) for child, _move in get_neighbors(parent)
                        }
                        for child in edges.get(parent, []):
                            if child not in solvable or child.boxes == parent.boxes:
                                continue
                            moved = next(
                                pos
                                for label, pos in child.boxes
                                if (label, pos) not in parent.boxes
                            )
                            checked_pushes += 1
                            if any(
                                pos != moved
                                and abs(pos[0] - moved[0]) + abs(pos[1] - moved[1]) == 1
                                for _label, pos in child.boxes
                            ):
                                checked_adjacent_groups += 1
                            self.assertFalse(
                                creates_frozen_component_deadlock(
                                    child.boxes,
                                    child.board,
                                    moved,
                                )
                            )
                            self.assertIn((child.robot_pos, child.boxes), retained)

        self.assertGreater(solvable_layouts, 0)
        self.assertGreater(checked_pushes, 0)
        self.assertGreater(checked_adjacent_groups, 0)

    def test_apply_move_and_render_preserve_goals(self):
        state = parse_puzzle(["OOOOO", "O R O", "O A O", "O a O", "OOOOO"])
        final = apply_move(state, "Down")
        self.assertTrue(final.is_goal())
        self.assertIn("a", render_puzzle(state))

    def test_unsolvable_puzzle_terminates(self):
        puzzle = [
            "OOOOOO",
            "OA  aO",
            "O R  O",
            "OOOOOO",
        ]
        result = solve(puzzle)
        path, final, _, _ = result
        self.assertIsNone(path)
        self.assertIsNone(final)
        self.assertEqual(SearchStatus.PROVEN_UNSOLVABLE, result.status)

    def test_search_can_be_cancelled(self):
        state = parse_puzzle(["OOOOO", "O R O", "O A O", "O a O", "OOOOO"])
        cancelled = Event()
        cancelled.set()
        result = a_star_search(state, cancelled)
        path, final, _, visited = result
        self.assertIsNone(path)
        self.assertIsNone(final)
        self.assertEqual(0, visited)
        self.assertEqual(SearchStatus.CANCELLED, result.status)

    def test_push_neighbors_include_walk_to_push(self):
        state = parse_puzzle(
            [
                "OOOOOOO",
                "O  R  O",
                "O     O",
                "O  X SO",
                "OOOOOOO",
            ]
        )
        segments = [segment for _next_state, segment in get_push_neighbors(state)]
        self.assertIn(["Down", "Left", "Down", "Right"], segments)

    def test_forced_push_macro_preserves_the_complete_replay_path(self):
        state = parse_puzzle(["OOOOOOO", "OR X SO", "OOOOOOO"])
        first_state, first_segment = get_push_neighbors(state)[0]
        final, segment = collapse_forced_pushes(first_state, first_segment)

        replay = state
        for move in segment:
            replay = apply_move(replay, move)
        self.assertTrue(final.is_goal())
        self.assertEqual(final.boxes, replay.boxes)
        self.assertEqual(2, final.cost)
        self.assertEqual(["Right", "Right", "Right"], segment)

    def test_push_beam_solves_with_forced_macros_and_honors_state_budget(self):
        state = parse_puzzle(["OOOOOOO", "OR X SO", "OOOOOOO"])
        path, final, _elapsed, visited = push_beam_search(state, beam_width=4)
        self.assertTrue(final.is_goal())
        self.assertEqual(["Right", "Right", "Right"], [move for move, _ in path])
        self.assertLessEqual(visited, 2)

        branching = parse_puzzle(
            [
                "OOOOOOO",
                "O SS  O",
                "O     O",
                "O XXR O",
                "O     O",
                "OOOOOOO",
            ]
        )
        path, final, _elapsed, visited = push_beam_search(
            branching,
            beam_width=4,
            max_visited=1,
        )
        self.assertIsNone(path)
        self.assertIsNone(final)
        self.assertEqual(1, visited)

    def test_exact_push_search_matches_independent_step_reference(self):
        cases = [
            ["OOOOO", "O R O", "O X O", "O S O", "OOOOO"],
            ["OOOOOOO", "O R   O", "O XX  O", "O SS  O", "OOOOOOO"],
            ["OOOOOOO", "O R   O", "O AB  O", "O ab  O", "OOOOOOO"],
            ["OOOOOO", "OX R O", "O   SO", "OOOOOO"],
            ["OOOOOOO", "O S S O", "O X X O", "O  R  O", "OOOOOOO"],
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                initial = parse_puzzle(rows)
                expected = self._reference_min_pushes(initial)
                result = push_a_star_search(initial)
                if expected is None:
                    self.assertEqual(SearchStatus.PROVEN_UNSOLVABLE, result.status)
                    self.assertIsNone(result.path)
                else:
                    self.assertEqual(SearchStatus.SOLVED, result.status)
                    self.assertEqual(expected, result.final.cost)

    def test_solution_validation_rejects_corrupt_public_result(self):
        initial = parse_puzzle(["OOOOO", "O R O", "O X O", "O S O", "OOOOO"])
        from Searches.Sokomind import _validated_solution_result

        result = _validated_solution_result(
            initial,
            [("Left", (1, 1))],
            0.0,
            0,
        )
        self.assertEqual(SearchStatus.FAILED, result.status)
        self.assertEqual("incomplete-solution-path", result.reason)

    def test_push_astar_returns_replayable_step_path(self):
        state = parse_puzzle(
            [
                "OOOOOOO",
                "O  R  O",
                "O     O",
                "O  X SO",
                "OOOOOOO",
            ]
        )
        path, final, _, _ = push_a_star_search(state)
        self.assertEqual(["Down", "Left", "Down", "Right", "Right"], [move for move, _ in path])
        self.assertTrue(final.is_goal())

    def test_push_astar_optimizes_pushes_when_robot_positions_are_canonicalized(self):
        state = parse_puzzle(
            [
                "OOOOOOO",
                "O     O",
                "O  X  O",
                "O   R O",
                "OO    O",
                "O OO SO",
                "OOOOOOO",
            ]
        )
        path, final, _, _ = push_a_star_search(state)

        replay = state
        pushes = 0
        for move, _position in path:
            before = replay.boxes
            replay = apply_move(replay, move)
            pushes += replay.boxes != before

        self.assertTrue(final.is_goal())
        self.assertEqual(5, pushes)
        self.assertEqual(pushes, final.cost)

    def test_cli_reports_output_write_failures_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "missing" / "solution.txt"
            with self.assertLogs(level="ERROR") as logs, redirect_stdout(io.StringIO()):
                result = main(
                    [
                        "--puzzle",
                        "ultra-tiny",
                        "--algorithm",
                        "astar",
                        "--output",
                        str(output),
                    ]
                )
        self.assertEqual(2, result)
        self.assertIn("Could not write solution", logs.output[0])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
