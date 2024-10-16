import time
import sys
from collections import deque

# Directions for movement (Up, Down, Left, Right)
directions = {
    'Up': (-1, 0),  # Up
    'Down': (1, 0),   # Down
    'Right': (0, 1),   # Right
    'Left': (0, -1)   # Left
}

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos2[1] - pos1[1])

def is_box_trapped(box_pos, walls, storages):
    """Determine if a box is trapped in an irrecoverable position."""
    y, x = box_pos
    
    # Check if the box is in a corner with two adjacent walls (can't move at all)
    if ((y-1, x) in walls and (y, x-1) in walls) or \
       ((y-1, x) in walls and (y, x+1) in walls) or \
       ((y+1, x) in walls and (y, x-1) in walls) or \
       ((y+1, x) in walls and (y, x+1) in walls):
        return True  # Box is stuck in a corner (two adjacent walls)

    # Now, check if the box is along a wall, but can still move left or right
    if (y-1, x) in walls and (y+1, x) in walls:  # Box between two vertical walls
        if (x-1, x) not in walls and (x+1, x) not in walls:
            return False  # Box can still move left or right
    
    if (x-1, y) in walls and (x+1, y) in walls:  # Box between two horizontal walls
        if (y-1, y) not in walls and (y+1, y) not in walls:
            return False  # Box can still move up or down
    
    # Check if the box is in a "dead zone" (isolated from storage)
    if box_pos not in storages:
        # Additional logic to detect if it's stuck in a path where it can no longer move to storage
        return True

    return False  # The box is not trapped

def is_unpromising_state(state):
    """Check if the current state is unpromising, e.g., if a box is trapped."""
    # More conservative pruning: skip this for now or only apply basic logic
    # For instance, if a box is completely stuck in a corner with no possibility of moving
    return False  # Disable advanced pruning for now

class State:
    def __init__(self, robot_pos, boxes, box_targets, walls, storages, storage_targets, puzzle, cost=0):
        self.robot_pos = robot_pos
        self.boxes = boxes  # dict mapping box labels to positions
        self.box_targets = box_targets  # dict mapping box labels to target positions
        self.walls = walls
        self.storages = storages
        self.storage_targets = storage_targets  # dict mapping storage positions to box labels
        self.puzzle = puzzle  # Track the current puzzle structure
        self.cost = cost  # Cost to reach this state

    def is_goal(self):
        """Check if all boxes are in their target positions."""
        for label, pos in self.boxes.items():
            if label in self.box_targets and pos != self.box_targets[label]:
                return False
            if label.startswith('X') and pos not in self.storages:
                return False
        return True

    def __hash__(self):
        return hash((self.robot_pos, frozenset(self.boxes.items())))

    def __eq__(self, other):
        return self.robot_pos == other.robot_pos and self.boxes == other.boxes

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

def get_neighbors(state):
    """Generate valid neighboring states from the current state."""
    neighbors = []
    rows, cols = len(state.puzzle), len(state.puzzle[0])
    for move, (dy, dx) in directions.items():
        new_robot_y, new_robot_x = state.robot_pos[0] + dy, state.robot_pos[1] + dx
        new_robot_pos = (new_robot_y, new_robot_x)

        if not (0 <= new_robot_y < rows and 0 <= new_robot_x < cols) or new_robot_pos in state.walls:
            continue

        # Check if moving into a box
        box_at_new_pos = next((label for label, pos in state.boxes.items() if pos == new_robot_pos), None)
        if box_at_new_pos:
            new_box_y = new_robot_y + dy
            new_box_x = new_robot_x + dx
            new_box_pos = (new_box_y, new_box_x)

            if not (0 <= new_box_y < rows and 0 <= new_box_x < cols) or new_box_pos in state.walls or new_box_pos in state.boxes.values():
                continue

            new_boxes = state.boxes.copy()
            new_boxes[box_at_new_pos] = new_box_pos
            neighbors.append(State(new_robot_pos, new_boxes, state.box_targets, state.walls, state.storages, state.storage_targets, state.puzzle))

        else:
            neighbors.append(State(new_robot_pos, state.boxes, state.box_targets, state.walls, state.storages, state.storage_targets, state.puzzle))

    return neighbors

def bfs_search(initial_state):
    """Perform Breadth-First Search with no depth limit or pruning."""
    visited = set()
    queue = deque([(initial_state, [])])  # (state, path)

    start_time = time.perf_counter()
    sys.stdout.write("Searching... (0 seconds)\r")
    sys.stdout.flush()

    while queue:
        elapsed_time = time.perf_counter() - start_time
        sys.stdout.write(f"Searching... ({elapsed_time:.6f} seconds)\r")
        sys.stdout.flush()

        current_state, path = queue.popleft()

        if current_state in visited:
            continue

        visited.add(current_state)

        if current_state.is_goal():
            return path, current_state, elapsed_time

        for neighbor in get_neighbors(current_state):
            if neighbor not in visited:
                move_direction = get_move_direction(current_state.robot_pos, neighbor.robot_pos)
                new_path = path + [(neighbor.robot_pos, move_direction)]
                queue.append((neighbor, new_path))

    return None, None, elapsed_time

def get_move_direction(from_pos, to_pos):
    """Return the movement direction as a string."""
    dy, dx = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
    return 'Up' if dy == -1 else 'Down' if dy == 1 else 'Right' if dx == 1 else 'Left' if dx == -1 else 'unknown'

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
    solution, final_state, elapsed_time = bfs_search(initial_state)

    if solution:
        print(f"\nSolution found! Time: {elapsed_time:.6f} seconds")
        print("Solution path:\n")

        current_state = initial_state
        for i, (new_robot_pos, direction) in enumerate(solution):
            print(f"Move {i+1}: {direction} ({new_robot_pos})")
            current_state.robot_pos = new_robot_pos

            box_at_pos = next((label for label, pos in current_state.boxes.items() if pos == new_robot_pos), None)
            if box_at_pos:
                dy, dx = directions[direction]
                new_box_pos = (new_robot_pos[0] + dy, new_robot_pos[1] + dx)
                current_state.boxes[box_at_pos] = new_box_pos

            print_puzzle(current_state)

        print(f"Moves to solve the puzzle ({len(solution)}):")
        formatted_moves = [f"{move[1]}({move[0]})" for move in solution]
        print(" --> ".join(formatted_moves))
    else:
        print("\nPuzzle is not solvable.")

if __name__ == "__main__":
    main()
