import time
import sys
import argparse
import logging
from heapdict import heapdict
from collections import deque
from functools import lru_cache
from itertools import product
from scipy.optimize import linear_sum_assignment
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

# Directions for movement with their corresponding coordinate changes
DIRECTIONS = {
    'Up': (-1, 0),
    'Down': (1, 0),
    'Right': (0, 1),
    'Left': (0, -1)
}

@lru_cache(maxsize=None)
def manhattan_distance(pos1, pos2):
    """
    Calculate the Manhattan distance between two positions.

    Args:
        pos1 (tuple): (y, x) coordinates of the first position.
        pos2 (tuple): (y, x) coordinates of the second position.

    Returns:
        int: Manhattan distance.
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def is_corner(pos, walls):
    """
    Check if a given position is a corner (deadlock) based on surrounding walls.

    Args:
        pos (tuple): (y, x) coordinates of the position.
        walls (set): Set of positions representing walls.

    Returns:
        bool: True if the position is a corner, False otherwise.
    """
    y, x = pos
    # Check all four possible corner configurations
    return (
        ((y-1, x) in walls and (y, x-1) in walls) or
        ((y-1, x) in walls and (y, x+1) in walls) or
        ((y+1, x) in walls and (y, x-1) in walls) or
        ((y+1, x) in walls and (y, x+1) in walls)
    )

def is_deadlock(box_pos, walls, goals, box_label):
    """
    Determine if a box is in a deadlock position.

    Args:
        box_pos (tuple): (y, x) coordinates of the box.
        walls (set): Set of positions representing walls.
        goals (dict): Mapping of box labels to their goal positions.
        box_label (str): Label of the box.

    Returns:
        bool: True if the box is in a deadlock position, False otherwise.
    """
    if box_label != 'X' and box_pos not in goals.get(box_label, set()):
        # Dedicated box not on its goal
        return is_corner(box_pos, walls)
    elif box_label == 'X' and box_pos not in goals.get('X', set()):
        # Generic box not on any generic storage
        return is_corner(box_pos, walls)
    return False

def minimum_matching(boxes, goals):
    """
    Compute the minimum total Manhattan distance between boxes and their goals using the Hungarian algorithm.

    Args:
        boxes (list): List of tuples (label, position) for each box.
        goals (dict): Mapping of box labels to sets of goal positions.

    Returns:
        int: Total minimum cost. Returns infinity if no valid matching exists.
    """
    # Separate generic and dedicated boxes
    generic_boxes = [pos for label, pos in boxes if label == 'X']
    dedicated_boxes = [(label, pos) for label, pos in boxes if label != 'X']

    total_cost = 0

    # Handle dedicated boxes: each has exactly one dedicated goal
    for label, box_pos in dedicated_boxes:
        if label in goals and goals[label]:
            goal_pos = next(iter(goals[label]))
            total_cost += manhattan_distance(box_pos, goal_pos)
        else:
            # No dedicated goal found for this box
            return float('inf')

    # Handle generic boxes with generic goals
    if generic_boxes and goals.get('X'):
        cost_matrix = []
        for box in generic_boxes:
            cost_row = [manhattan_distance(box, goal) for goal in goals['X']]
            cost_matrix.append(cost_row)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for row, col in zip(row_ind, col_ind):
            total_cost += cost_matrix[row][col]
    elif generic_boxes or (goals.get('X') and len(goals['X']) < len(generic_boxes)):
        # Mismatch in number of generic boxes and generic goals
        return float('inf')

    return total_cost

class State:
    """
    Represents the state of the puzzle, including the robot position, box positions, and cost metrics.
    """
    __slots__ = ('robot_pos', 'boxes', 'cost', 'heuristic', 'priority')

    # Class variables shared across all instances
    walls = frozenset()
    generic_storages = frozenset()
    dedicated_storages = {}
    puzzle = []
    width = 0
    height = 0

    def __init__(self, robot_pos, boxes, cost=0):
        """
        Initialize a State instance.

        Args:
            robot_pos (tuple): (y, x) coordinates of the robot.
            boxes (list): List of tuples (label, position) for each box.
            cost (int, optional): Cost to reach this state. Defaults to 0.
        """
        self.robot_pos = robot_pos
        # Boxes are stored as sorted tuples of (label, position) for immutability and hashing
        self.boxes = tuple(sorted(boxes, key=lambda x: (x[0], x[1])))
        self.cost = cost
        self.heuristic = 0
        self.priority = 0

    def calculate_heuristic(self, algorithm='astar'):
        """
        Calculate the heuristic value based on the sum of minimum Manhattan distances.

        Args:
            algorithm (str, optional): The search algorithm being used. Defaults to 'astar'.
        """
        if algorithm in ['astar', 'greedy']:
            # Create a goals dictionary based on current box labels
            self.heuristic = minimum_matching(self.boxes, {
                label: self.dedicated_storages[label] if label != 'X' else self.generic_storages
                for label in set(label for label, _ in self.boxes)
            })
            if algorithm == 'astar':
                self.priority = self.cost + self.heuristic
            else:
                self.priority = self.heuristic
        else:
            self.heuristic = 0
            self.priority = self.cost

    def is_goal(self):
        """
        Check if all boxes are in their target positions.

        Returns:
            bool: True if the state is a goal state, False otherwise.
        """
        for label, pos in self.boxes:
            if label == 'X':
                if pos not in self.generic_storages:
                    return False
            else:
                if pos not in self.dedicated_storages.get(label, set()):
                    return False
        return True

    def __hash__(self):
        """
        Compute the hash of the state for use in sets and dictionaries.

        Returns:
            int: Hash value.
        """
        return hash((self.robot_pos, self.boxes))

    def __eq__(self, other):
        """
        Check equality between two states.

        Args:
            other (State): Another state to compare with.

        Returns:
            bool: True if both states are identical, False otherwise.
        """
        return self.robot_pos == other.robot_pos and self.boxes == other.boxes

    def __lt__(self, other):
        """
        Compare states based on their priority for priority queues.

        Args:
            other (State): Another state to compare with.

        Returns:
            bool: True if this state has lower priority, False otherwise.
        """
        return self.priority < other.priority

def parse_puzzle(puzzle):
    """
    Parse the puzzle layout and initialize the initial state.

    Args:
        puzzle (list): List of strings representing the puzzle grid.

    Returns:
        State: The initial state of the puzzle.
    """
    robot_pos = None
    boxes = []
    walls = set()
    generic_storages = set()
    dedicated_storages = {}

    for y, row in enumerate(puzzle):
        for x, char in enumerate(row):
            pos = (y, x)
            if char == 'R':
                robot_pos = pos
            elif char == 'O':
                walls.add(pos)
            elif char == 'S':
                generic_storages.add(pos)
            elif char.islower() and char not in {'o', 'r', 's', 'x'}:
                # Dedicated storage identified by lowercase letters
                dedicated_storages.setdefault(char.upper(), set()).add(pos)
            elif char == 'X':
                boxes.append(('X', pos))
            elif char.isupper() and char not in {'O', 'R', 'S', 'X'}:
                # Dedicated box identified by uppercase letters
                boxes.append((char, pos))

    # Validate that robot position exists
    if robot_pos is None:
        logging.error("No robot position ('R') found in the puzzle.")
        sys.exit(1)

    # Validate that each dedicated box has at least one dedicated storage
    for label, _ in boxes:
        if label != 'X' and label not in dedicated_storages:
            logging.error(f"No dedicated storage found for box '{label}'.")
            sys.exit(1)

    # Set class variables for shared puzzle elements
    State.walls = frozenset(walls)
    State.generic_storages = frozenset(generic_storages)
    State.dedicated_storages = dedicated_storages
    State.puzzle = puzzle
    State.height = len(puzzle)
    State.width = max(len(row) for row in puzzle)

    return State(robot_pos, boxes)

def print_puzzle(state):
    """
    Print the current puzzle layout based on the state of the robot and boxes.

    Args:
        state (State): The current state of the puzzle.
    """
    # Create a mutable copy of the puzzle layout
    puzzle_layout = [list(row) for row in State.puzzle]

    # Reset all movable elements to empty spaces except walls and storages
    for y, row in enumerate(puzzle_layout):
        for x, char in enumerate(row):
            if char in {'O', 'S'} or (char.islower() and char not in {'o', 'r', 's', 'x'}):
                # Retain walls, generic storages, and dedicated storages
                continue
            else:
                puzzle_layout[y][x] = ' '

    # Place the boxes in their current positions with colors
    for label, box in state.boxes:
        y, x = box
        if label == 'X':
            puzzle_layout[y][x] = Fore.YELLOW + label + Style.RESET_ALL
        else:
            puzzle_layout[y][x] = Fore.GREEN + label + Style.RESET_ALL

    # Place the robot in its current position with a distinct color
    ry, rx = state.robot_pos
    puzzle_layout[ry][rx] = Fore.CYAN + 'R' + Style.RESET_ALL

    # Print the puzzle grid
    for row in puzzle_layout:
        print(' '.join(row))
    print("\n")

def get_move_direction(from_pos, to_pos):
    """
    Determine the movement direction based on the change in position.

    Args:
        from_pos (tuple): (y, x) starting position.
        to_pos (tuple): (y, x) ending position.

    Returns:
        str: Direction of movement ('Up', 'Down', 'Left', 'Right'), or 'unknown' if not found.
    """
    dy, dx = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
    for direction, (ddy, ddx) in DIRECTIONS.items():
        if (dy, dx) == (ddy, ddx):
            return direction
    return 'unknown'

def get_neighbors(state, algorithm='astar'):
    """
    Generate all valid neighboring states from the current state.

    Args:
        state (State): The current state.
        algorithm (str, optional): The search algorithm being used. Defaults to 'astar'.

    Returns:
        list: List of tuples containing neighboring states and the move direction.
    """
    neighbors = []
    ry, rx = state.robot_pos

    for move, (dy, dx) in DIRECTIONS.items():
        new_robot_y, new_robot_x = ry + dy, rx + dx
        new_robot_pos = (new_robot_y, new_robot_x)

        # Check if new robot position is within bounds and not a wall
        if not (0 <= new_robot_y < State.height and 0 <= new_robot_x < State.width):
            logging.debug(f"Move '{move}' leads out of bounds to {new_robot_pos}. Skipping.")
            continue
        if new_robot_pos in State.walls:
            logging.debug(f"Move '{move}' leads into a wall at {new_robot_pos}. Skipping.")
            continue

        # Check if the robot is moving into a box
        box_found = False
        box_label = None
        for label, pos in state.boxes:
            if pos == new_robot_pos:
                box_found = True
                box_label = label
                break

        if box_found:
            # Calculate the new box position after being pushed
            new_box_y, new_box_x = new_robot_y + dy, new_robot_x + dx
            new_box_pos = (new_box_y, new_box_x)

            # Validate new box position
            if not (0 <= new_box_y < State.height and 0 <= new_box_x < State.width):
                logging.debug(f"Pushing box '{box_label}' {move} leads out of bounds to {new_box_pos}. Skipping.")
                continue
            if new_box_pos in State.walls:
                logging.debug(f"Pushing box '{box_label}' {move} leads into a wall at {new_box_pos}. Skipping.")
                continue
            # Check if new_box_pos is occupied by another box
            if any(pos == new_box_pos for _, pos in state.boxes):
                logging.debug(f"Pushing box '{box_label}' {move} leads into another box at {new_box_pos}. Skipping.")
                continue
            # Check for deadlock
            if is_deadlock(new_box_pos, State.walls, {
                label: State.dedicated_storages[label] if label != 'X' else State.generic_storages
                for label in set(label for label, _ in state.boxes)
            }, box_label):
                logging.debug(f"Pushing box '{box_label}' {move} to {new_box_pos} results in deadlock. Skipping.")
                continue

            # Generate new boxes tuple with the pushed box moved
            new_boxes = []
            for label, pos in state.boxes:
                if pos == new_robot_pos:
                    new_boxes.append((label, new_box_pos))
                else:
                    new_boxes.append((label, pos))
            new_boxes = tuple(sorted(new_boxes, key=lambda x: (x[0], x[1])))

            # Create a new neighbor state with updated positions and cost
            neighbor = State(new_robot_pos, new_boxes, state.cost + 1)
            neighbor.calculate_heuristic(algorithm)
            neighbors.append((neighbor, move))
            logging.debug(f"Pushing box '{box_label}' {move} to {new_box_pos} resulting in new state.")
        else:
            # Normal movement without pushing a box
            neighbor = State(new_robot_pos, state.boxes, state.cost + 1)
            neighbor.calculate_heuristic(algorithm)
            neighbors.append((neighbor, move))
            logging.debug(f"Moving robot {move} to {new_robot_pos} without pushing any box.")

    return neighbors

def reconstruct_path(came_from, end_state):
    """
    Reconstruct the path from the initial state to the end state.

    Args:
        came_from (dict): Mapping of states to their parent states and the move taken.
        end_state (State): The goal state.

    Returns:
        list: List of tuples containing move directions and robot positions.
    """
    path = []
    current = end_state
    while current in came_from:
        parent, move = came_from[current]
        path.append((move, current.robot_pos))
        current = parent
    path.reverse()
    return path

def dfs_search(initial_state):
    """
    Perform Depth-First Search to explore the state space.

    Args:
        initial_state (State): The starting state.

    Returns:
        tuple: (solution path, final state, elapsed time, states visited)
    """
    visited = set()          # Set to keep track of visited states
    stack = [initial_state]  # Stack for DFS
    came_from = {}           # Dictionary to reconstruct the path
    states_visited = 0      # Counter for states visited

    start_time = time.perf_counter()
    logging.info("Searching using Depth-First Search (DFS)...")

    while stack:
        elapsed_time = time.perf_counter() - start_time
        logging.debug(f"Elapsed Time: {elapsed_time:.6f} seconds | States Visited: {states_visited}")

        current_state = stack.pop()
        states_visited += 1

        if current_state in visited:
            continue

        visited.add(current_state)

        if current_state.is_goal():
            logging.info("Goal state reached!")
            path = reconstruct_path(came_from, current_state)
            return path, current_state, elapsed_time, states_visited

        for neighbor, move_dir in get_neighbors(current_state, 'dfs'):
            if neighbor not in visited:
                if neighbor not in came_from:
                    came_from[neighbor] = (current_state, move_dir)
                stack.append(neighbor)

    return None, None, elapsed_time, states_visited

def bfs_search(initial_state):
    """
    Perform Breadth-First Search to explore the state space.

    Args:
        initial_state (State): The starting state.

    Returns:
        tuple: (solution path, final state, elapsed time, states visited)
    """
    visited = set()              # Set to keep track of visited states
    queue = deque([initial_state])  # Queue for BFS
    came_from = {}               # Dictionary to reconstruct the path
    states_visited = 0          # Counter for states visited

    start_time = time.perf_counter()
    logging.info("Searching using Breadth-First Search (BFS)...")

    while queue:
        elapsed_time = time.perf_counter() - start_time
        logging.debug(f"Elapsed Time: {elapsed_time:.6f} seconds | States Visited: {states_visited}")

        current_state = queue.popleft()
        states_visited += 1

        if current_state in visited:
            continue

        visited.add(current_state)

        if current_state.is_goal():
            logging.info("Goal state reached!")
            path = reconstruct_path(came_from, current_state)
            return path, current_state, elapsed_time, states_visited

        for neighbor, move_dir in get_neighbors(current_state, 'bfs'):
            if neighbor not in visited:
                if neighbor not in came_from:
                    came_from[neighbor] = (current_state, move_dir)
                queue.append(neighbor)

    return None, None, elapsed_time, states_visited

def greedy_search(initial_state):
    """
    Perform Greedy Best-First Search using a heuristic to guide exploration.

    Args:
        initial_state (State): The starting state.

    Returns:
        tuple: (solution path, final state, elapsed time, states visited)
    """
    visited = set()                   # Set to keep track of visited states
    priority_queue = heapdict()       # Priority queue based on heuristic
    initial_state.calculate_heuristic('greedy')
    priority_queue[initial_state] = initial_state.priority
    came_from = {}                    # Dictionary to reconstruct the path
    states_visited = 0                # Counter for states visited

    start_time = time.perf_counter()
    logging.info("Searching using Greedy Best-First Search...")

    while priority_queue:
        elapsed_time = time.perf_counter() - start_time
        logging.debug(f"Elapsed Time: {elapsed_time:.6f} seconds | States Visited: {states_visited}")

        current_state, _ = priority_queue.popitem()
        states_visited += 1

        if current_state in visited:
            continue

        visited.add(current_state)

        if current_state.is_goal():
            logging.info("Goal state reached!")
            path = reconstruct_path(came_from, current_state)
            return path, current_state, elapsed_time, states_visited

        for neighbor, move_dir in get_neighbors(current_state, 'greedy'):
            if neighbor not in visited:
                if neighbor not in came_from or neighbor.priority < initial_state.priority:
                    came_from[neighbor] = (current_state, move_dir)
                    priority_queue[neighbor] = neighbor.priority

    return None, None, elapsed_time, states_visited  # No solution found

def a_star_search(initial_state):
    """
    Perform A* Search using both cost and heuristic to guide exploration.

    Args:
        initial_state (State): The starting state.

    Returns:
        tuple: (solution path, final state, elapsed time, states visited)
    """
    visited = {}                     # Dictionary to keep track of the lowest cost to reach each state
    priority_queue = heapdict()      # Priority queue based on total cost (f = g + h)
    initial_state.calculate_heuristic('astar')
    priority_queue[initial_state] = initial_state.priority
    came_from = {}                   # Dictionary to reconstruct the path
    states_visited = 0               # Counter for states visited

    start_time = time.perf_counter()
    logging.info("Searching using A* Search...")

    while priority_queue:
        elapsed_time = time.perf_counter() - start_time
        logging.debug(f"Elapsed Time: {elapsed_time:.6f} seconds | States Visited: {states_visited}")

        current_state, _ = priority_queue.popitem()
        states_visited += 1

        # Skip if we've already found a better path to this state
        if current_state in visited and visited[current_state] <= current_state.cost:
            continue

        visited[current_state] = current_state.cost

        if current_state.is_goal():
            logging.info("Goal state reached!")
            path = reconstruct_path(came_from, current_state)
            return path, current_state, elapsed_time, states_visited

        for neighbor, move_dir in get_neighbors(current_state, 'astar'):
            new_cost = current_state.cost + 1  # Assuming each move has a cost of 1
            if neighbor in visited and visited[neighbor] <= new_cost:
                continue
            if neighbor not in came_from or new_cost < visited.get(neighbor, float('inf')):
                came_from[neighbor] = (current_state, move_dir)
                neighbor.cost = new_cost
                neighbor.calculate_heuristic('astar')
                priority_queue[neighbor] = neighbor.priority

    return None, None, elapsed_time, states_visited  # No solution found

def load_custom_puzzle(file_path):
    """
    Load a custom puzzle from a file.

    Args:
        file_path (str): Path to the puzzle file.

    Returns:
        list: List of strings representing the puzzle grid.
    """
    try:
        with open(file_path, 'r') as file:
            puzzle = [line.strip('\n') for line in file if line.strip()]
        return puzzle
    except FileNotFoundError:
        logging.error(f"Puzzle file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading puzzle file: {e}")
        sys.exit(1)

def setup_logging_level():
    """
    Prompt the user to select a logging level.

    Returns:
        str: Selected logging level.
    """
    logging_levels = {
        '1': 'DEBUG',
        '2': 'INFO',
        '3': 'ERROR'
    }
    print("\nSelect a logging level:")
    print("1. DEBUG")
    print("2. INFO (default)")
    print("3. ERROR")

    while True:
        choice = input("Enter the number corresponding to your choice: ").strip()
        if choice in logging_levels:
            return logging_levels[choice]
        elif choice == '':
            return 'INFO'
        else:
            print("Invalid input. Please try again.")

def setup_logging(selected_level):
    """
    Configure the logging settings based on the selected level.

    Args:
        selected_level (str): The selected logging level.
    """
    level = getattr(logging, selected_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("puzzle_solver.log", mode='w')
        ]
    )
    logging.info(f"Logging level set to {selected_level.upper()}.")

def interactive_puzzle_selection(puzzles):
    """
    Interactively prompt the user to select a puzzle.

    Args:
        puzzles (dict): Available puzzles.

    Returns:
        tuple: (puzzle_name, puzzle)
    """
    puzzle_options = list(puzzles.keys())
    puzzle_options.append('End program')
    print("\nSelect a puzzle size:")
    for idx, option in enumerate(puzzle_options, 1):
        print(f"{idx}. {option}")

    while True:
        try:
            puzzle_choice = int(input("Enter the number corresponding to your choice: "))
            if puzzle_choice == len(puzzle_options):
                print("Exiting the program.")
                sys.exit(0)
            puzzle_name = puzzle_options[puzzle_choice - 1]
            puzzle = puzzles[puzzle_name]
            return puzzle_name, puzzle
        except (ValueError, IndexError):
            print("Invalid input. Please try again.")

def interactive_algorithm_selection():
    """
    Interactively prompt the user to select a search algorithm.

    Returns:
        str: Selected algorithm.
    """
    search_algorithms = ['DFS', 'BFS', 'Greedy', 'A*', 'End program']
    print("\nSelect a search algorithm:")
    for idx, algo in enumerate(search_algorithms, 1):
        print(f"{idx}. {algo}")

    while True:
        try:
            algo_choice = int(input("Enter the number corresponding to your choice: "))
            if algo_choice == len(search_algorithms):
                print("Exiting the program.")
                sys.exit(0)
            algorithm = search_algorithms[algo_choice - 1]
            return algorithm
        except (ValueError, IndexError):
            print("Invalid input. Please try again.")

def confirm_selections(puzzle_name, algorithm, logging_level):
    """
    Confirm the user's selections before starting the search.

    Args:
        puzzle_name (str): Name of the selected puzzle.
        algorithm (str): Selected search algorithm.
        logging_level (str): Selected logging level.

    Returns:
        bool: True if the user confirms, False otherwise.
    """
    print("\nYou have selected the following options:")
    print(f"Puzzle: {puzzle_name}")
    print(f"Search Algorithm: {algorithm}")
    print(f"Logging Level: {logging_level}")
    confirmation = input("Are you ready to begin? (y/n): ").strip().lower()
    return confirmation == 'y'

def export_solution(solution):
    """
    Prompt the user to export the solution to a file.

    Args:
        solution (list): List of tuples containing move directions and robot positions.
    """
    export = input("Do you want to export the solution to a file? (y/n): ").strip().lower()
    if export == 'y':
        export_path = input("Enter the file path to save the solution: ").strip()
        try:
            with open(export_path, 'w') as f:
                for i, (direction, pos) in enumerate(solution, 1):
                    f.write(f"Move {i}: {direction} ({pos})\n")
            logging.info(f"Solution successfully exported to {export_path}")
        except Exception as e:
            logging.error(f"Failed to export solution: {e}")

def main():
    """
    Main function to run the puzzle solver. Handles user interaction for selecting puzzles and algorithms.
    """
    # Define the available puzzles excluding 'Huge'
    puzzles = {
        'Ultra Tiny': [
            "OOOOO",
            "O R O",
            "O A O",
            "O a O",
            "OOOOO"
        ],
        'Tiny': [
            "OOOOOO",
            "O R  O",
            "O XO O",
            "OO A O",
            "OSa  O",
            "OOOOOO"
        ],
        'Medium': [
            "OOOOOOO",
            "Oa   bO",
            "O AXB O",
            "O XRX O",
            "OSCXDSO",
            "OcS SdO",
            "OOOOOOO"
        ],
        'Large': [
            "OOOOOOOOOO",
            "OOOOOOOSSO",
            "OOOOO  abO",
            "OOOOO XSSO",
            "OOOOOO  OO",
            "OR     OOO",
            "OO A X X O",
            "OO BXO O O",
            "OO   O   O",
            "OOOOOOOOOO"
        ]
    }

    # Prompt the user to select a puzzle
    puzzle_name, puzzle = interactive_puzzle_selection(puzzles)

    # Prompt the user to select a search algorithm
    algorithm = interactive_algorithm_selection()

    # Prompt the user to select a logging level
    logging_level = setup_logging_level()

    # Set up logging based on the selected level
    setup_logging(logging_level)

    # Confirm the user's selections
    if not confirm_selections(puzzle_name, algorithm, logging_level):
        print("Exiting the program.")
        sys.exit(0)

    logging.info(f"\nYou have selected {algorithm} search on the {puzzle_name} puzzle.\n")

    # Display the selected puzzle with colors
    logging.info("Puzzle:")
    print_puzzle(parse_puzzle(puzzle))  # Temporarily parse to display; will parse again below

    # Initialize the puzzle state
    initial_state = parse_puzzle(puzzle)

    # Execute the selected search algorithm
    if algorithm == 'DFS':
        solution, final_state, elapsed_time, states_visited = dfs_search(initial_state)
    elif algorithm == 'BFS':
        solution, final_state, elapsed_time, states_visited = bfs_search(initial_state)
    elif algorithm == 'Greedy':
        solution, final_state, elapsed_time, states_visited = greedy_search(initial_state)
    elif algorithm == 'A*':
        solution, final_state, elapsed_time, states_visited = a_star_search(initial_state)
    else:
        logging.error("Invalid algorithm selected.")
        sys.exit(1)

    # Process and display the solution if found
    if solution:
        logging.info("\nSolution found!")
        current_state = initial_state
        for i, (direction, new_robot_pos) in enumerate(solution):
            logging.info(f"Move {i+1}: {direction} ({new_robot_pos})")
            current_state.robot_pos = new_robot_pos

            # Update box positions if a box was pushed
            for label, pos in current_state.boxes:
                if pos == new_robot_pos:
                    dy, dx = DIRECTIONS[direction]
                    new_box_pos = (new_robot_pos[0] + dy, new_robot_pos[1] + dx)
                    # Update the box position
                    new_boxes = []
                    for lbl, p in current_state.boxes:
                        if lbl == label and p == pos:
                            new_boxes.append((lbl, new_box_pos))
                        else:
                            new_boxes.append((lbl, p))
                    current_state.boxes = tuple(sorted(new_boxes, key=lambda x: (x[0], x[1])))
                    break

            # Display the updated puzzle layout
            print_puzzle(current_state)

        # Display metrics: time taken and states visited
        logging.info(f"Time taken: {elapsed_time:.6f} seconds")
        logging.info(f"States visited: {states_visited}")
        logging.info(f"Moves to solve the puzzle ({len(solution)}):")
        formatted_moves = [f"{move}({pos})" for move, pos in solution]
        logging.info(" --> ".join(formatted_moves))

        # Prompt to export the solution
        export_solution(solution)
    else:
        # Inform the user if the puzzle is unsolvable
        logging.info("\nPuzzle is not solvable.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Handle user interruption gracefully
        logging.info("\nProgram terminated by user.")
        sys.exit(0)
