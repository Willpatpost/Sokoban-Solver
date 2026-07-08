"""Compatibility entry point for A* search."""
import sys
try:
    from .Sokomind import *  # noqa: F401,F403
except ImportError:
    from Sokomind import *  # noqa: F401,F403

if __name__ == "__main__":
    raise SystemExit(main(["--algorithm", "astar", *sys.argv[1:]]))
