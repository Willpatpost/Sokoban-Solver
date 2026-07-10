import unittest
from threading import Event

from Searches.Sokomind import (
    BUILTIN_PUZZLES,
    PuzzleError,
    apply_move,
    a_star_search,
    get_push_neighbors,
    get_neighbors,
    parse_puzzle,
    push_a_star_search,
    render_puzzle,
    solve,
)


class SokomindTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
