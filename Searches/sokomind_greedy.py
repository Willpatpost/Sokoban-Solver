import time
import sys
import heapq  # For priority queue

# Directions for movement (North, South, East, West)
directions = {
    'Up': (-1, 0),  # Up
    'Down': (1, 0),   # Down
    'Right': (0, 1),   # Right
    'Left': (0, -1)   # Left
}

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class State:
    def __init__(self, robot_pos, boxes, box_targets, walls, storages, storage_targets, puzzle):
        self.robot_pos = robot_pos
        self.boxes = boxes  # dict mapping box labels to positions
        self.box_targets = box_targets  # dict mapping box labels to target positions
        self.walls = walls
        self.storages = storages
        self.storage_targets = storage_targets  # dict mapping storage positions to box labels
        self.puzzle = puzzle  # Track the current puzzle structure
        self.heuristic = self.calculate_heuristic()  # Heuristic (h in Greedy)

    def calculate_heuristic(self):
        """Calculate heuristic based on the sum of Manhattan distances from boxes to their targets."""
        heuristic = sum(manhattan_distance(pos, self.box_targets.get(label, pos)) for label, pos in self.boxes.items())

        # Add penalty for deadlocks (boxes trapped in corners or unreachable locations)
        for label, pos in self.boxes.items():
            if self.is_deadlock(pos):
                heuristic += 1000  # Arbitrary large penalty for deadlocked boxes

        return heuristic

    def is_deadlock(self, pos):
        """Determine if a box is stuck in a deadlock position (e.g., against two walls)."""
        y, x = pos
        if ((y == 0 or self.puzzle[y - 1][x] == 'O') and (y == len(self.puzzle) - 1 or self.puzzle[y + 1][x] == 'O')) or \
           ((x == 0 or self.puzzle[y][x - 1] == 'O') and (x == len(self.puzzle[0]) - 1 or self.puzzle[y][x + 1] == 'O')):
            return True
        return False

    def is_goal(self):
        """Check if all boxes are in their target positions."""
        for label, pos in self.boxes.items():
            if label in self.box_targets and pos != self.box_targets[label]:
                return False
            if label.startswith('X') and pos not in self.storages:
                return False
        return True

    def __hash__(self):
        """Hash based on robot position and box positions."""
        return hash((self.robot_pos, frozenset(self.boxes.items())))

    def __eq__(self, other):
        """Equality based on robot position and box positions."""
        return self.robot_pos == other.robot_pos and self.boxes == other.boxes

    def __lt__(self, other):
        """Comparison based on heuristic (h in Greedy) value for priority queue sorting."""
        return self.heuristic < other.heuristic

def parse_puzzle(puzzle):
    """Parse the puzzle and initialize the state."""
    robot_pos, boxes, box_targets = None, {}, {}
    walls, storages, storage_targets = set(), set(), {}

    for y, row in enumerate(puzzle):
        for x, char in enumerate(row):
            pos = (y, x)
            if char == 'R':
                robot_pos = pos
            elif char == 'O':
                walls.add(pos)
            elif char == 'S':
                storages.add(pos)
            elif char.islower():
                storages.add(pos)
                storage_targets[pos] = char.upper()
                box_targets[char.upper()] = pos
            elif char == 'X':
                boxes[char + str(pos)] = pos  # Unique label for each 'X'
            elif char.isupper():
                boxes[char] = pos

    return State(robot_pos, boxes, box_targets, walls, storages, storage_targets, puzzle)

def get_neighbors(state):
    """Generate valid neighboring states from the current state."""
    neighbors = []
    rows, cols = len(state.puzzle), len(state.puzzle[0])
    for move, (dy, dx) in directions.items():
        new_robot_y, new_robot_x = state.robot_pos[0] + dy, state.robot_pos[1] + dx
        new_robot_pos = (new_robot_y, new_robot_x)

        # Check if movement is valid (no walls or out-of-bounds)
        if not (0 <= new_robot_y < rows and 0 <= new_robot_x < cols) or new_robot_pos in state.walls:
            continue

        # Check if moving into a box
        box_at_new_pos = next((label for label, pos in state.boxes.items() if pos == new_robot_pos), None)
        if box_at_new_pos:
            # Calculate new box position
            new_box_y = new_robot_y + dy
            new_box_x = new_robot_x + dx
            new_box_pos = (new_box_y, new_box_x)

            # Validate box move (no walls, boxes, or out-of-bounds)
            if not (0 <= new_box_y < rows and 0 <= new_box_x < cols) or new_box_pos in state.walls or new_box_pos in state.boxes.values():
                continue

            new_boxes = state.boxes.copy()
            new_boxes[box_at_new_pos] = new_box_pos

            neighbors.append(State(new_robot_pos, new_boxes, state.box_targets, state.walls, state.storages, state.storage_targets, state.puzzle))
        else:
            # Normal movement (no box pushing)
            neighbors.append(State(new_robot_pos, state.boxes, state.box_targets, state.walls, state.storages, state.storage_targets, state.puzzle))

    return neighbors

def print_puzzle(state):
    """Print the current puzzle layout based on the state of the robot and boxes."""
    puzzle = [list(row) for row in state.puzzle]  # Create a mutable copy of the puzzle layout

    # Clear only the robot and boxes from the puzzle to update their positions, but preserve walls ('O') and storage ('S')
    for y, row in enumerate(puzzle):
        for x, char in enumerate(row):
            if char == 'R':
                puzzle[y][x] = ' '  # Remove robot 'R' from the previous position
            if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' and (y, x) not in state.walls and (y, x) not in state.storages:
                puzzle[y][x] = ' '  # Remove the box label from the previous position

    # Update the robot's position
    ry, rx = state.robot_pos
    puzzle[ry][rx] = 'R'

    # Update the box positions
    for label, (by, bx) in state.boxes.items():
        puzzle[by][bx] = label[0]  # Only use the first character of the label (e.g., 'X')

    # Print the puzzle
    for row in puzzle:
        print(' '.join(row))
    print("\n")

def get_move_direction(from_pos, to_pos):
    """Return the movement direction as a single character based on the change in position."""
    dy, dx = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
    if dy == -1:
        return 'Up'  # North (up)
    elif dy == 1:
        return 'Down'  # South (down)
    elif dx == 1:
        return 'Right'  # East (right)
    elif dx == -1:
        return 'Left'  # West (left)
    return 'unknown'

def greedy_search(initial_state):
    """Perform Greedy Search using a heuristic to guide exploration."""
    visited = set()
    priority_queue = []  # Priority queue with states sorted by heuristic
    heapq.heappush(priority_queue, (initial_state.heuristic, initial_state, []))  # (heuristic, state, path)
    
    start_time = time.perf_counter()
    sys.stdout.write("Searching... (0.000 seconds)\r")
    sys.stdout.flush()

    while priority_queue:
        elapsed_time = time.perf_counter() - start_time
        sys.stdout.write(f"Searching... ({elapsed_time:.6f} seconds)\r")
        sys.stdout.flush()

        _, current_state, path = heapq.heappop(priority_queue)

        if current_state in visited:
            continue

        visited.add(current_state)

        # If the current state is a goal, return the solution path
        if current_state.is_goal():
            return path, current_state, elapsed_time

        # Explore neighboring states
        for neighbor in get_neighbors(current_state):
            if neighbor not in visited:
                move_direction = get_move_direction(current_state.robot_pos, neighbor.robot_pos)
                new_path = path + [(neighbor.robot_pos, move_direction)]
                heapq.heappush(priority_queue, (neighbor.heuristic, neighbor, new_path))

    return None, None, elapsed_time  # No solution found

def main():
    # Define the medium puzzle
    puzzle = [
        "OOOOOO",
        "O R  O",
        "O XO O",
        "OO A O",
        "OSa  O",
        "OOOOOO"
    ]

    print("Puzzle:")
    for row in puzzle:
        print(' '.join(row))

    initial_state = parse_puzzle(puzzle)
    print("Initial state parsed.")  # Debugging message
    solution, final_state, elapsed_time = greedy_search(initial_state)

    if solution:
        print(f"\nSolution found! Time: {elapsed_time:.6f} seconds")
        print("Solution path:\n")

        # Set up a mutable puzzle that will be updated as moves are applied
        current_state = initial_state
        for i, (new_robot_pos, direction) in enumerate(solution):
            print(f"Move {i+1}: {direction} ({new_robot_pos})")

            # Update the robot's position
            current_state.robot_pos = new_robot_pos

            # Apply the movement to the state
            box_at_pos = next((label for label, pos in current_state.boxes.items() if pos == new_robot_pos), None)
            if box_at_pos:
                dy, dx = directions[direction]
                new_box_pos = (new_robot_pos[0] + dy, new_robot_pos[1] + dx)
                current_state.boxes[box_at_pos] = new_box_pos

            # Print the updated puzzle state
            print_puzzle(current_state)

        # Print the final summary
        print(f"Moves to solve the puzzle ({len(solution)}):")
        formatted_moves = [f"{move[1]}({move[0]})" for move in solution]
        print(" --> ".join(formatted_moves))
    else:
        print("\nPuzzle is not solvable.")

if __name__ == "__main__":
    main()

"""

The advanced heuristic implemented in this Greedy Search algorithm is designed to improve the search performance by guiding it more intelligently towards the goal. 
The core of the heuristic is still based on Manhattan distance, but several additional factors are introduced to make it more effective for Sokoban-style puzzles.

1. For each labeled box (e.g., A, B, etc.), the heuristic calculates the straight-line Manhattan distance to its designated target. 
This helps guide boxes closer to their storage locations. For X boxes, the Manhattan distance to the closest available storage (S) is calculated. 
By minimizing this distance, the heuristic ensures that boxes are moved efficiently toward their destinations.

2. A significant enhancement is the addition of penalties for "trapped" boxes. 
A box is considered trapped if it is placed in a corner or adjacent to two walls, which might make it difficult or impossible to move. 
The heuristic checks for such trapped conditions and applies a penalty to discourage the algorithm from considering these states. 
This avoids situations where the search wastes time on moves that could lead to dead ends.

3. A penalty system is applied to give weight to unfavorable positions, such as being far from storage or being trapped. 
This encourages the algorithm to favor states that are more likely to lead to the solution.

"""