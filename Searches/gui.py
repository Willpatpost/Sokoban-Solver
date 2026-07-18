"""Tkinter desktop interface for playing and watching Sokomind."""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from .Sokomind import (
        BUILTIN_PUZZLES,
        DIRECTIONS,
        PuzzleError,
        State,
        a_star_search,
        apply_move,
        bfs_search,
        dfs_search,
        greedy_search,
        load_custom_puzzle,
        parse_puzzle,
        portfolio_search,
        push_a_star_search,
        push_greedy_search,
        ultimate_search,
        weighted_push_a_star_search,
    )
except ImportError:
    from Sokomind import (  # type: ignore
        BUILTIN_PUZZLES,
        DIRECTIONS,
        PuzzleError,
        State,
        a_star_search,
        apply_move,
        bfs_search,
        dfs_search,
        greedy_search,
        load_custom_puzzle,
        parse_puzzle,
        portfolio_search,
        push_a_star_search,
        push_greedy_search,
        ultimate_search,
        weighted_push_a_star_search,
    )


SEARCHES = {
    "Ultimate Search": ultimate_search,
    "Fast Portfolio": portfolio_search,
    "Fast Push A*": weighted_push_a_star_search,
    "Push A*": push_a_star_search,
    "Push Greedy": push_greedy_search,
    "A*": a_star_search,
    "Greedy": greedy_search,
    "BFS": bfs_search,
    "DFS": dfs_search,
}

KEY_MOVES = {
    "Up": "Up", "Down": "Down", "Left": "Left", "Right": "Right",
    "w": "Up", "W": "Up", "s": "Down", "S": "Down",
    "a": "Left", "A": "Left", "d": "Right", "D": "Right",
}


class SokomindApp(tk.Tk):
    TILE_MIN = 28
    TILE_MAX = 72
    ANIMATION_MS = 110

    def __init__(self) -> None:
        super().__init__()
        self.title("Sokomind")
        self.geometry("900x720")
        self.minsize(620, 500)

        self.state: State
        self.initial_state: State
        self.history: list[State] = []
        self.moves = 0
        self._animation: list[str] = []
        self._job_id = 0
        self._cancel_event: threading.Event | None = None
        self._results: queue.Queue = queue.Queue()
        self._custom_puzzle: list[str] | None = None
        self._level_cards: dict[str, tk.Frame] = {}
        self._completion_window: tk.Toplevel | None = None
        self._completion_shown = False
        self._timer_started_at: float | None = None
        self._timer_elapsed = 0.0
        self._timer_job: str | None = None

        self.puzzle_name = tk.StringVar(value="ultra-tiny")
        self.algorithm = tk.StringVar(value="Ultimate Search")
        self.status = tk.StringVar(value="Ready")
        self.move_text = tk.StringVar(value="Moves: 0")
        self.timer_text = tk.StringVar(value="Time: 00:00")

        self._build_ui()
        self.load_selected_puzzle()
        self._show_home()
        self.bind_all("<KeyPress>", self._on_key)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(50, self._poll_results)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=10)
        toolbar.pack(fill="x")

        ttk.Label(toolbar, text="Solver").pack(side="left")
        ttk.Combobox(
            toolbar, textvariable=self.algorithm,
            values=list(SEARCHES), state="readonly", width=16,
        ).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Solve", command=self.solve_animated).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Hint", command=self.hint).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Stop", command=self.stop).pack(side="left", padx=2)

        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(toolbar, text="Undo", command=self.undo).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Reset", command=self.reset).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Home", command=self._show_home).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Open custom...", command=self.open_puzzle).pack(
            side="right", padx=2
        )

        body = ttk.Frame(self, padding=(10, 0, 10, 8))
        body.pack(fill="both", expand=True)
        sidebar = tk.Frame(body, width=190, background="#202631")
        sidebar.pack(side="left", fill="y", padx=(0, 10))
        sidebar.pack_propagate(False)
        tk.Label(
            sidebar, text="LEVELS", background="#202631", foreground="#aab4c3",
            font=("Segoe UI", 10, "bold"), anchor="w", padx=12, pady=10,
        ).pack(fill="x")
        self._level_list = tk.Frame(sidebar, background="#202631")
        self._level_list.pack(fill="both", expand=True)
        for name, puzzle in BUILTIN_PUZZLES.items():
            self._add_level_card(name, puzzle)

        self.canvas = tk.Canvas(body, background="#171a21", highlightthickness=0)
        self.canvas.pack(side="right", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw())

        footer = ttk.Frame(self, padding=(10, 3, 10, 10))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status).pack(side="left")
        ttk.Label(footer, text="Arrow keys or WASD to play").pack(side="left", padx=24)
        ttk.Label(footer, textvariable=self.move_text).pack(side="right")
        ttk.Label(footer, textvariable=self.timer_text).pack(side="right", padx=18)

        self.home = tk.Frame(self, background="#111722")
        hero = tk.Frame(self.home, background="#111722")
        hero.place(relx=.5, rely=.48, anchor="center")
        tk.Label(
            hero, text="S", width=3, background="#397fc3", foreground="white",
            font=("Segoe UI", 30, "bold"),
        ).pack(pady=(0, 12))
        tk.Label(
            hero, text="Sokomind", background="#111722", foreground="white",
            font=("Segoe UI", 30, "bold"),
        ).pack()
        tk.Label(
            hero, text="Think ahead. Push with purpose.", background="#111722",
            foreground="#91a0b3", font=("Segoe UI", 12),
        ).pack(pady=(3, 22))
        directions = (
            "Push every box onto its matching goal.\n\n"
            "Arrow keys / WASD   Move\n"
            "Backspace / U       Undo\n"
            "R                   Reset\n\n"
            "X boxes belong on S goals. Lettered boxes belong on\n"
            "the matching lowercase goal. Boxes can be pushed, not pulled."
        )
        tk.Label(
            hero, text=directions, justify="left", background="#1d2633",
            foreground="#dbe6f2", font=("Segoe UI", 11), padx=24, pady=18,
        ).pack()
        ttk.Button(hero, text="Play Sokomind", command=self._hide_home).pack(
            pady=(22, 0), ipadx=24, ipady=6
        )

    def _add_level_card(self, name: str, puzzle: list[str]) -> None:
        card = tk.Frame(
            self._level_list, background="#2a313d", highlightthickness=2,
            highlightbackground="#2a313d", cursor="hand2",
        )
        card.pack(fill="x", padx=8, pady=5)
        preview = tk.Canvas(
            card, width=74, height=60, background="#171a21",
            highlightthickness=0, cursor="hand2",
        )
        preview.pack(side="left", padx=6, pady=6)
        title = tk.Label(
            card, text=name.replace("-", " ").title(), background="#2a313d",
            foreground="white", font=("Segoe UI", 10, "bold"), cursor="hand2",
            anchor="w",
        )
        title.pack(side="left", fill="x", expand=True, padx=(2, 6))
        self._draw_thumbnail(preview, puzzle)
        for widget in (card, preview, title):
            widget.bind("<Button-1>", lambda _event, level=name: self.select_level(level))
        self._level_cards[name] = card

    @staticmethod
    def _draw_thumbnail(canvas: tk.Canvas, puzzle: list[str]) -> None:
        width = max(map(len, puzzle))
        height = len(puzzle)
        tile = max(2, min(9, 68 // width, 54 // height))
        ox = (74 - tile * width) // 2
        oy = (60 - tile * height) // 2
        colors = {
            "O": "#586475", "R": "#54a8e8", "X": "#d88d3c",
            "S": "#d39b31", " ": "#252a34",
        }
        for y, row in enumerate(puzzle):
            for x, char in enumerate(row):
                color = colors.get(
                    char, "#d88d3c" if char.isupper() else "#d39b31"
                )
                canvas.create_rectangle(
                    ox + x * tile, oy + y * tile,
                    ox + (x + 1) * tile, oy + (y + 1) * tile,
                    fill=color, outline="#303744", width=1,
                )

    def select_level(self, name: str) -> None:
        self.puzzle_name.set(name)
        self.load_selected_puzzle()

    def _update_level_selection(self) -> None:
        selected = self.puzzle_name.get()
        for name, card in self._level_cards.items():
            card.configure(
                highlightbackground="#54a8e8" if name == selected else "#2a313d"
            )

    def _set_state(self, state: State, *, record: bool = True) -> None:
        if record:
            self.history.append(self.state)
        self.state = state
        self._start_timer()
        self.moves += 1
        self.move_text.set(f"Moves: {self.moves}")
        self.draw()
        if state.is_goal():
            self.stop(keep_status=True)
            self._freeze_timer()
            self.status.set(f"Solved in {self.moves} moves!")
            if not self._completion_shown:
                self._completion_shown = True
                self.after_idle(self._show_completion)

    def load_selected_puzzle(self) -> None:
        self._custom_puzzle = None
        self._load(BUILTIN_PUZZLES[self.puzzle_name.get()])
        self._update_level_selection()

    def open_puzzle(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Sokomind puzzle",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            puzzle = load_custom_puzzle(Path(path))
        except PuzzleError as exc:
            messagebox.showerror("Invalid puzzle", str(exc), parent=self)
            return
        if not self._load(puzzle):
            return
        self._custom_puzzle = puzzle
        self.puzzle_name.set(Path(path).name)
        self._update_level_selection()

    def _load(self, puzzle: list[str]) -> bool:
        try:
            state = parse_puzzle(puzzle)
        except PuzzleError as exc:
            messagebox.showerror("Invalid puzzle", str(exc), parent=self)
            return False
        self.stop()
        self.initial_state = state
        self.state = state
        self.history.clear()
        self.moves = 0
        self._reset_timer()
        self._completion_shown = False
        self.move_text.set("Moves: 0")
        self.status.set("Ready - use arrow keys or WASD")
        self.draw()
        return True

    def reset(self) -> None:
        self.stop()
        self.state = self.initial_state
        self.history.clear()
        self.moves = 0
        self._reset_timer()
        self._completion_shown = False
        self.move_text.set("Moves: 0")
        self.status.set("Reset")
        self.draw()

    def _show_home(self) -> None:
        self.stop(keep_status=True)
        self._freeze_timer()
        self.home.place(x=0, y=0, relwidth=1, relheight=1)
        self.home.lift()

    def _hide_home(self) -> None:
        self.home.place_forget()
        self.canvas.focus_set()

    def _start_timer(self) -> None:
        if self._timer_started_at is None:
            self._timer_started_at = time.perf_counter() - self._timer_elapsed
            self._tick_timer()

    def _tick_timer(self) -> None:
        if self._timer_started_at is None:
            return
        self._timer_elapsed = time.perf_counter() - self._timer_started_at
        minutes, seconds = divmod(int(self._timer_elapsed), 60)
        self.timer_text.set(f"Time: {minutes:02d}:{seconds:02d}")
        self._timer_job = self.after(250, self._tick_timer)

    def _freeze_timer(self) -> None:
        if self._timer_started_at is not None:
            self._timer_elapsed = time.perf_counter() - self._timer_started_at
            self._timer_started_at = None
        if self._timer_job is not None:
            self.after_cancel(self._timer_job)
            self._timer_job = None

    def _reset_timer(self) -> None:
        self._freeze_timer()
        self._timer_elapsed = 0.0
        self.timer_text.set("Time: 00:00")

    def _show_completion(self) -> None:
        if self._completion_window is not None and self._completion_window.winfo_exists():
            return
        dialog = tk.Toplevel(self)
        self._completion_window = dialog
        dialog.title("Level passed")
        dialog.configure(background="#202631")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        tk.Label(
            dialog, text="\u2605", background="#202631", foreground="#f4c761",
            font=("Segoe UI Symbol", 42),
        ).pack(pady=(18, 0))
        tk.Label(
            dialog, text="Congratulations!", background="#202631",
            foreground="white", font=("Segoe UI", 20, "bold"),
        ).pack(padx=42)
        tk.Label(
            dialog, text="Level passed", background="#202631",
            foreground="#aab4c3", font=("Segoe UI", 12),
        ).pack(pady=(2, 4))
        tk.Label(
            dialog, text=f"Completed in {self.moves} moves",
            background="#202631", foreground="#65b96e", font=("Segoe UI", 11, "bold"),
        ).pack(pady=(0, 16))
        buttons = ttk.Frame(dialog, padding=(14, 0, 14, 16))
        buttons.pack()
        ttk.Button(buttons, text="Replay", command=self._replay_from_dialog).pack(
            side="left", padx=4
        )
        names = list(BUILTIN_PUZZLES)
        current = self.puzzle_name.get()
        if current in names and names.index(current) + 1 < len(names):
            ttk.Button(buttons, text="Next level", command=self._next_from_dialog).pack(
                side="left", padx=4
            )
        ttk.Button(buttons, text="Keep looking", command=dialog.destroy).pack(
            side="left", padx=4
        )
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - dialog.winfo_width()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

    def _replay_from_dialog(self) -> None:
        if self._completion_window is not None:
            self._completion_window.destroy()
        self.reset()

    def _next_from_dialog(self) -> None:
        if self._completion_window is not None:
            self._completion_window.destroy()
        names = list(BUILTIN_PUZZLES)
        current = self.puzzle_name.get()
        if current in names and names.index(current) + 1 < len(names):
            self.select_level(names[names.index(current) + 1])

    def undo(self) -> None:
        self.stop()
        if not self.history:
            return
        self.state = self.history.pop()
        self.moves = max(0, self.moves - 1)
        self._completion_shown = False
        self.move_text.set(f"Moves: {self.moves}")
        self.status.set("Undid one move")
        self.draw()

    def _on_key(self, event: tk.Event) -> None:
        if self.home.winfo_ismapped():
            return
        if isinstance(self.focus_get(), ttk.Combobox):
            return
        move = KEY_MOVES.get(event.keysym) or KEY_MOVES.get(event.char)
        if move:
            self.stop()
            self.try_move(move)
        elif event.keysym in {"BackSpace", "u"}:
            self.undo()
        elif event.keysym == "r":
            self.reset()

    def try_move(self, move: str) -> bool:
        try:
            next_state = apply_move(self.state, move)
        except ValueError:
            self.status.set(f"{move} is blocked")
            return False
        self._set_state(next_state)
        if not next_state.is_goal():
            self.status.set("Playing")
        return True

    def _start_solver(self, purpose: str) -> None:
        self.stop()
        snapshot = self.state
        algorithm = self.algorithm.get()
        self._job_id += 1
        job_id = self._job_id
        cancel_event = threading.Event()
        self._cancel_event = cancel_event
        self.status.set(f"{algorithm} is searching...")

        def run() -> None:
            result = SEARCHES[algorithm](snapshot, cancel_event)
            self._results.put((job_id, purpose, snapshot, algorithm, result))

        threading.Thread(target=run, daemon=True, name="sokomind-solver").start()

    def solve_animated(self) -> None:
        self._start_solver("solve")

    def hint(self) -> None:
        self._start_solver("hint")

    def stop(self, keep_status: bool = False) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
            self._cancel_event = None
        self._job_id += 1
        self._animation.clear()
        if not keep_status and hasattr(self, "state"):
            self.status.set("Stopped")

    def _poll_results(self) -> None:
        try:
            while True:
                job_id, purpose, snapshot, algorithm, result = self._results.get_nowait()
                if job_id != self._job_id or snapshot != self.state:
                    continue
                path, _final, elapsed, visited = result
                if path is None:
                    suffix = (
                        " in current search budget"
                        if algorithm in {"Ultimate Search", "Fast Portfolio"}
                        else ""
                    )
                    self.status.set(f"{algorithm}: no solution{suffix} ({visited:,} states)")
                    continue
                moves = [move for move, _position in path]
                self.status.set(
                    f"{algorithm}: {len(moves)} moves, {visited:,} states, {elapsed:.2f}s"
                )
                if purpose == "hint":
                    if moves:
                        self.status.set(f"Hint: {moves[0]}  -  {len(moves)} moves remain")
                else:
                    self._animation = moves
                    self.after(self.ANIMATION_MS, self._animate_next)
        except queue.Empty:
            pass
        self.after(50, self._poll_results)

    def _animate_next(self) -> None:
        if not self._animation or self.state.is_goal():
            return
        move = self._animation.pop(0)
        if self.try_move(move) and self._animation:
            self.after(self.ANIMATION_MS, self._animate_next)

    def draw(self) -> None:
        if not hasattr(self, "state"):
            return
        self.canvas.delete("all")
        board = self.state.board
        height = len(board.rows)
        width = len(board.rows[0])
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        tile = max(
            self.TILE_MIN,
            min(self.TILE_MAX, (canvas_width - 24) // width, (canvas_height - 24) // height),
        )
        origin_x = (canvas_width - tile * width) // 2
        origin_y = (canvas_height - tile * height) // 2
        boxes = {pos: label for label, pos in self.state.boxes}

        for y, row in enumerate(board.rows):
            for x, char in enumerate(row):
                x1, y1 = origin_x + x * tile, origin_y + y * tile
                x2, y2 = x1 + tile, y1 + tile
                pos = (y, x)
                if pos in board.walls:
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2, fill="#394352", outline="#566274", width=2
                    )
                    continue
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill="#252a34", outline="#303744"
                )
                goal_label = None
                if pos in board.generic_goals:
                    goal_label = "S"
                else:
                    for label, goals in board.dedicated_goals:
                        if pos in goals:
                            goal_label = label.lower()
                            break
                if goal_label:
                    margin = tile * 0.31
                    self.canvas.create_oval(
                        x1 + margin, y1 + margin, x2 - margin, y2 - margin,
                        fill="#d39b31", outline="#f4c761", width=2,
                    )
                    self.canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        text=goal_label, fill="#2b2112",
                        font=("Segoe UI", max(9, tile // 5), "bold"),
                    )
                if pos in boxes:
                    label = boxes[pos]
                    margin = tile * 0.12
                    on_goal = pos in board.goals_for(label)
                    self.canvas.create_rectangle(
                        x1 + margin, y1 + margin, x2 - margin, y2 - margin,
                        fill="#65b96e" if on_goal else "#d88d3c",
                        outline="#a9e0ae" if on_goal else "#f2b568", width=3,
                    )
                    self.canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        text=label, fill="#171a21",
                        font=("Segoe UI", max(11, tile // 3), "bold"),
                    )
                if pos == self.state.robot_pos:
                    margin = tile * 0.15
                    self.canvas.create_oval(
                        x1 + margin, y1 + margin, x2 - margin, y2 - margin,
                        fill="#54a8e8", outline="#a6d8ff", width=3,
                    )
                    self.canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        text="R", fill="white",
                        font=("Segoe UI", max(11, tile // 3), "bold"),
                    )

    def _close(self) -> None:
        self._job_id += 1
        self._freeze_timer()
        self.destroy()


def main() -> int:
    try:
        app = SokomindApp()
        app.mainloop()
    except tk.TclError as exc:
        print(f"Could not start the GUI: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
