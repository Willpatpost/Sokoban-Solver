import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from threading import Event

from Searches.Sokomind import (
    BUILTIN_PUZZLES,
    CONFORMANCE_PATH,
    HEURISTIC_CACHE_SIZE,
    PuzzleError,
    _heuristic_for_layout,
    _minimum_assignment_cost,
    _reachable_parents,
    _reconstruct_walk,
    apply_move,
    a_star_search,
    get_push_neighbors,
    get_neighbors,
    main,
    parse_puzzle,
    push_a_star_search,
    render_puzzle,
    solve,
)


class SokomindTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conformance = json.loads(CONFORMANCE_PATH.read_text(encoding="utf-8"))

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
                self.assertEqual(expected["legalMoves"], sorted(
                    move for _next_state, move in get_neighbors(state)
                ))
                self.assertEqual(expected["mechanicalMoves"], sorted(
                    move for _next_state, move in get_neighbors(
                        state, prune_deadlocks=False
                    )
                ))
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
            with self.subTest(case=case["id"]), self.assertRaisesRegex(
                PuzzleError, patterns[case["errorKind"]]
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
        state = parse_puzzle([
            "OOOOOOO",
            "O  R  O",
            "O     O",
            "O  X SO",
            "OOOOOOO",
        ])
        parents = _reachable_parents(state)
        self.assertEqual(["Down", "Left", "Down"], _reconstruct_walk(parents, (3, 2)))
        self.assertIsNone(parents[state.robot_pos])
        self.assertTrue(all(
            record is None or isinstance(record, tuple)
            for record in parents.values()
        ))

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
            with self.subTest(goal=reserved_goal), self.assertRaisesRegex(
                PuzzleError, "Unsupported symbol"
            ):
                parse_puzzle(puzzle)

    def test_static_dead_square_push_is_pruned(self):
        state = parse_puzzle([
            "OOOOOO",
            "O    O",
            "O RX O",
            "O  S O",
            "OOOOOO",
        ])
        # Pushing right strands the box against the right wall, away from its goal.
        moves = [move for _, move in get_neighbors(state)]
        self.assertNotIn("Right", moves)

    def test_player_can_make_legal_push_into_dead_square(self):
        state = parse_puzzle([
            "OOOOOO",
            "O    O",
            "O RX O",
            "O  S O",
            "OOOOOO",
        ])
        pushed = apply_move(state, "Right")
        self.assertIn(("X", (2, 4)), pushed.boxes)

    def test_static_2x2_box_deadlock_push_is_pruned(self):
        state = parse_puzzle([
            "OOOOOO",
            "O    O",
            "O  XXO",
            "ORX OO",
            "O SSSO",
            "OOOOOO",
        ])
        moves = [move for _, move in get_neighbors(state)]
        self.assertNotIn("Right", moves)

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
        path, final, _, _ = solve(puzzle)
        self.assertIsNone(path)
        self.assertIsNone(final)

    def test_search_can_be_cancelled(self):
        state = parse_puzzle(["OOOOO", "O R O", "O A O", "O a O", "OOOOO"])
        cancelled = Event()
        cancelled.set()
        path, final, _, visited = a_star_search(state, cancelled)
        self.assertIsNone(path)
        self.assertIsNone(final)
        self.assertEqual(0, visited)

    def test_push_neighbors_include_walk_to_push(self):
        state = parse_puzzle([
            "OOOOOOO",
            "O  R  O",
            "O     O",
            "O  X SO",
            "OOOOOOO",
        ])
        segments = [segment for _next_state, segment in get_push_neighbors(state)]
        self.assertIn(["Down", "Left", "Down", "Right"], segments)

    def test_push_astar_returns_replayable_step_path(self):
        state = parse_puzzle([
            "OOOOOOO",
            "O  R  O",
            "O     O",
            "O  X SO",
            "OOOOOOO",
        ])
        path, final, _, _ = push_a_star_search(state)
        self.assertEqual(["Down", "Left", "Down", "Right", "Right"], [move for move, _ in path])
        self.assertTrue(final.is_goal())

    def test_push_astar_optimizes_pushes_when_robot_positions_are_canonicalized(self):
        state = parse_puzzle([
            "OOOOOOO",
            "O     O",
            "O  X  O",
            "O   R O",
            "OO    O",
            "O OO SO",
            "OOOOOOO",
        ])
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
                result = main([
                    "--puzzle", "ultra-tiny",
                    "--algorithm", "astar",
                    "--output", str(output),
                ])
        self.assertEqual(2, result)
        self.assertIn("Could not write solution", logs.output[0])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
