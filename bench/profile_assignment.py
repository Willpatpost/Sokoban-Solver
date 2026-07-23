"""Profile full Hungarian assignment against safe one-row repair."""

from __future__ import annotations

import json
import sys
import statistics
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from searches.Sokomind import _minimum_assignment, _repair_minimum_assignment


def matrix(size: int, offset: int) -> list[list[float]]:
    return [
        [float(((row + 3) * (column + 5) + offset * 7) % 31) for column in range(size)]
        for row in range(size)
    ]


def timed(operation, iterations: int) -> float:
    started = time.perf_counter()
    for index in range(iterations):
        operation(index)
    return (time.perf_counter() - started) * 1000


def main() -> None:
    results = []
    for size in range(2, 11):
        base = matrix(size, 0)
        previous = _minimum_assignment(base)
        iterations = max(500, 12_000 // size)
        full_samples = []
        repair_samples = []
        for repeat in range(7):

            def full(index: int) -> None:
                changed = [row.copy() for row in base]
                changed[index % size] = matrix(size, index + repeat)[index % size]
                _minimum_assignment(changed)

            def repair(index: int) -> None:
                changed = [row.copy() for row in base]
                changed_row = index % size
                changed[changed_row] = matrix(size, index + repeat)[changed_row]
                _repair_minimum_assignment(previous, changed, changed_row)

            full_samples.append(timed(full, iterations))
            repair_samples.append(timed(repair, iterations))
        full_ms = statistics.median(full_samples)
        repair_ms = statistics.median(repair_samples)
        results.append(
            {
                "size": size,
                "iterations": iterations,
                "fullMs": round(full_ms, 3),
                "repairMs": round(repair_ms, 3),
                "ratio": round(repair_ms / full_ms, 3),
            }
        )
    print(json.dumps({"runtime": "python", "results": results}, indent=2))


if __name__ == "__main__":
    main()
