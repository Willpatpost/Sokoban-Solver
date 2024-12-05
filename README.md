# Sokomind

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)

## Overview

This project implements a solver capable of solving Sokomind puzzles using a variety of search algorithms, including Depth-First Search (DFS), Breadth-First Search (BFS), Greedy Best-First Search, and A* Search. The solver is designed to handle both generic and dedicated boxes, detect deadlocks, and optimize the path to the goal state using advanced heuristics.

## Features

- **Multiple Search Algorithms**: Supports DFS, BFS, Greedy Best-First Search, and A* Search to solve puzzles.
- **Customizable Heuristics**: Utilizes the Hungarian algorithm for optimal box-goal matching, enhancing heuristic accuracy over basic Manhattan distance.
- **Custom Puzzle Input**: Allows users to input their own puzzles via text files.
- **Colored Visualization**: Enhances puzzle readability with colored representations of robots, boxes, walls, and storages using the `colorama` library.
- **Detailed Logging**: Offers selectable logging levels (`DEBUG`, `INFO`, `ERROR`) to monitor the solver's execution and facilitate debugging.
- **Solution Export**: Enables exporting the solution path to a file for future reference.
- **Performance Optimizations**: Implements efficient state representation, caching mechanisms, and optimized heuristic calculations to speed up the solving process.
- **Comprehensive Error Handling**: Provides clear error messages and validates puzzle configurations before attempting to solve them.
- **Modular Code Structure**: Organized into distinct functions for better readability and maintainability.

## Requirements

- **Python 3.6 or higher**

### Python Libraries

- [heapdict](https://pypi.org/project/heapdict/) (`heapdict`)
- [scipy](https://www.scipy.org/) (`scipy`)
- [colorama](https://pypi.org/project/colorama/) (`colorama`)

### Installation of Python Libraries

You can install the required Python libraries using `pip`:

```bash
pip install heapdict scipy colorama
