# PathPilot

PathPilot is an interactive navigation game that will grow incrementally from a
manual 2D grid into a complete AI navigation and evaluation project.

## Current status

**Phase 1 — Milestone 3:** interactive game controls.

Implemented behavior:

- Configurable grid dimensions, start, goal, and obstacles
- Validated one-cell movement in four directions
- Collision and boundary handling
- Start and goal placement
- Obstacle placement and removal
- Agent-only reset and full-board reset
- Standard-library unit tests
- Pygame window with a centered, responsive grid
- Visual markers for the start, goal, and agent
- Modern control panel with active and hover states
- Click-and-drag obstacle painting and right-drag erasing
- Easy click-based start and goal placement
- Manual navigation with arrow keys or WASD
- Working Reset Agent and Clear Grid controls

### Controls

| Control | Action |
| --- | --- |
| Obstacles | Left-click/drag to paint; right-click/drag to erase |
| Set Start | Click a valid grid cell to move the start and agent |
| Set Goal | Click a valid grid cell to move the goal |
| Play | Move with arrow keys or WASD |
| Reset Agent | Return the agent to the current start |
| Clear Grid | Remove obstacles and restore the original endpoints |
| Keys 1–4 | Select Obstacles, Set Start, Set Goal, or Play |
| Escape | Close PathPilot |

After placing the start or goal, PathPilot automatically returns to Obstacles
mode to prevent accidental endpoint changes.

## Run the tests

No third-party packages are needed to test the environment:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Run the interface

Install the project and its sole runtime dependency, then launch it:

```bash
python -m pip install -e .
pathpilot
```

Alternatively, with the dependency already installed:

```bash
PYTHONPATH=src python -m pathpilot.game
```

Python 3.10 or newer is required. The project uses the actively maintained
`pygame-ce` distribution, imported in code through the standard `pygame` API.
Close the window using its standard close button or Escape.
