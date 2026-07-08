import unittest
from threading import Event

from Searches.Sokomind import (
    PuzzleError,
    apply_move,
    a_star_search,
    get_neighbors,
    parse_puzzle,
    render_puzzle,
    solve,
)


class SokomindTests(unittest.TestCase):
    def test_all_searches_solve_simple_puzzle(self):
        puzzle = ["OOOOO", "O R O", "O A O", "O a O", "OOOOO"]
        for algorithm in ("astar", "greedy", "bfs", "dfs"):
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


if __name__ == "__main__":
    unittest.main()
