Solves advanced sokoban puzzles created in txt files using BFS, DFS, Greedy, and A* searches.

The rules:

1. One box can be moved at a time.
2. Boxes can only be pushed by a robot. Boxes cannot be pulled.
3. Neither the robot nor the box can pass through obstacles, walls, or other boxes.
4. A robot cannot push more than one box.
5. The boxes marked in upper cases must be put in the storage with the corresponding lower cases.
6. The goal is achieved when all boxes are in their storage spots.

Key:

O = walls/obstacles

R = robot/user

S = generic storage (filled by any X)

X = generic box (goes to any S storage)

{a, b, ..., z} = storage for their corresponding uppercase boxes

{A, B, ..., Z} = boxes for their corresponding lowercase storages.
