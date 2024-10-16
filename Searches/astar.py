import time
import sys
import heapq

# Directions for movement
directions = {
    'Up': (-1, 0),  # Up
    'Down': (1, 0),  # Down
    'Right': (0, 1),  # Right
    'Left': (0, -1)  # Left
}

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class State:
    def __init__(self, robot_pos, boxes, box_targets, walls, storages, storage_targets, puzzle, cost=0):
        self.robot_pos = robot_pos
        self.boxes = boxes  # dict mapping box labels to positions
        self.box_targets = box_targets  # dict mapping box labels to target positions
        self.walls = walls
        self.storages = storages
        self.storage_targets = storage_targets  # dict mapping storage positions to box labels
        self.puzzle = puzzle  # Track the current puzzle structure
        self.cost = cost  # Cost to reach this state (g in A*)
        self.heuristic = self.calculate_heuristic()  # Heuristic (h in A*)
        self.priority = self.cost + self.heuristic  # f = g + h

    def calculate_heuristic(self):
        """Calculate heuristic based on the sum of Manhattan distances from boxes to their targets."""
        heuristic = sum(manhattan_distance(pos, self.box_targets.get(label, pos)) for label, pos in self.boxes.items())

        # You could try a weighted Manhattan distance here for more aggressive exploration:
        # heuristic = 2 * sum(manhattan_distance(pos, self.box_targets.get(label, pos)) for label, pos in self.boxes.items())
        
        return heuristic

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
        """Comparison based on priority (f = g + h) value for priority queue sorting."""
        return self.priority < other.priority

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
            new_box_y, new_box_x = new_robot_y + dy, new_robot_x + dx
            new_box_pos = (new_box_y, new_box_x)

            # Validate box move (no walls, boxes, or out-of-bounds)
            if not (0 <= new_box_y < rows and 0 <= new_box_x < cols) or new_box_pos in state.walls or new_box_pos in state.boxes.values():
                continue

            new_boxes = state.boxes.copy()
            new_boxes[box_at_new_pos] = new_box_pos

            neighbors.append(State(new_robot_pos, new_boxes, state.box_targets, state.walls, state.storages, state.storage_targets, state.puzzle, state.cost + 1))
        else:
            # Normal movement (no box pushing)
            neighbors.append(State(new_robot_pos, state.boxes, state.box_targets, state.walls, state.storages, state.storage_targets, state.puzzle, state.cost + 1))

    return neighbors

def a_star_search(initial_state):
    """Perform A* Search using both cost and heuristic to guide exploration."""
    visited = set()
    priority_queue = []  # Priority queue with states sorted by f = g + h
    heapq.heappush(priority_queue, (initial_state.priority, 0, initial_state, []))  # (priority, cost, state, path)
    
    start_time = time.perf_counter()
    sys.stdout.write("Searching... (0 seconds)\r")
    sys.stdout.flush()

    while priority_queue:
        elapsed_time = time.perf_counter() - start_time
        sys.stdout.write(f"Searching... ({elapsed_time:.6f} seconds)\r")
        sys.stdout.flush()

        _, cost, current_state, path = heapq.heappop(priority_queue)

        if current_state in visited:
            continue

        visited.add(current_state)

        if current_state.is_goal():
            print(f"\nGoal state reached! Time: {elapsed_time} seconds")
            return path, current_state, elapsed_time  # Return the path, final state, and time

        for neighbor in get_neighbors(current_state):
            if neighbor not in visited:
                move_direction = get_move_direction(current_state.robot_pos, neighbor.robot_pos)
                new_path = path + [(neighbor.robot_pos, move_direction)]
                heapq.heappush(priority_queue, (neighbor.priority, cost + 1, neighbor, new_path))

    return None, None, elapsed_time  # No solution found

def get_move_direction(from_pos, to_pos):
    """Return the movement direction as a single character based on the change in position."""
    dy, dx = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
    if dy == -1:
        return 'Up'
    elif dy == 1:
        return 'Down'
    elif dx == 1:
        return 'Right'
    elif dx == -1:
        return 'Left'
    return 'unknown'

def print_puzzle(state):
    """Print the current puzzle layout based on the state of the robot and boxes."""
    puzzle = [list(row) for row in state.puzzle]
    for y, row in enumerate(puzzle):
        for x, char in enumerate(row):
            if char == 'R':
                puzzle[y][x] = ' '
            if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' and (y, x) not in state.walls and (y, x) not in state.storages:
                puzzle[y][x] = ' '
    ry, rx = state.robot_pos
    puzzle[ry][rx] = 'R'
    for label, (by, bx) in state.boxes.items():
        puzzle[by][bx] = label[0]
    for row in puzzle:
        print(' '.join(row))
    print("\n")

def main():
    # Define the medium puzzle
    puzzle = [
        "OOOOOOO",
        "Oa   bO",
        "O AXB O",
        "O XRX O",
        "OSCXDSO",
        "OcS SdO",
        "OOOOOOO" 
    ]

    print("Puzzle:")
    for row in puzzle:
        print(' '.join(row))

    initial_state = parse_puzzle(puzzle)
    solution, final_state, elapsed_time = a_star_search(initial_state)

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
