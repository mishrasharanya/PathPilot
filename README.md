# PathPilot

PathPilot is an interactive navigation game that will grow incrementally from a
manual 2D grid into a complete AI navigation and evaluation project.

## Current status

**Phase 1 — Milestone 1:** UI-independent grid environment.

Implemented behavior:

- Configurable grid dimensions, start, goal, and obstacles
- Validated one-cell movement in four directions
- Collision and boundary handling
- Start and goal placement
- Obstacle placement and removal
- Agent-only reset and full-board reset
- Standard-library unit tests

The Pygame interface will be introduced in the next milestone only after this
environment API is approved.

## Run the tests

No third-party packages are needed to test the environment:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Python 3.10 or newer is required. Pygame is declared as the sole runtime
dependency for the upcoming graphical interface.
