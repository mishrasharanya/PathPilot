# PathPilot

PathPilot is an interactive navigation game that will grow incrementally from a
manual 2D grid into a complete AI navigation and evaluation project.

## Current status

**Phase 2 — Milestone 6:** reproducible evaluation and benchmarks.

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
- Animated A* exploration and final-path overlays
- Plan, pause/resume, and stop visualization controls
- Autonomous baseline that runs A* again before every movement
- Live planning calls, replans, nodes, moves, and planning latency
- Optimized navigation that caches a valid route and replans only when blocked
- Paired, seeded benchmarks across static and dynamic scenarios
- Raw CSV results and aggregate JSON summaries

### Controls

| Control | Action |
| --- | --- |
| Obstacles | Left-click to toggle; left-drag to paint; right-click/drag to erase |
| Set Start | Click a valid grid cell to move the start and agent |
| Set Goal | Click a valid grid cell to move the goal |
| Play | Move with arrow keys or WASD |
| Reset Agent | Return the agent to the current start |
| Clear Grid | Remove obstacles and restore the original endpoints |
| Keys 1–4 | Select Obstacles, Set Start, Set Goal, or Play |
| Escape | Close PathPilot |
| Plan A* | Search from the agent's current position to the goal |
| Baseline | Move automatically while replanning before every step |
| Cached | Follow a cached route and replan only when its remaining path is blocked |
| Pause / Resume | Pause or continue the visualization |
| Stop | Clear the current exploration and path overlay |

After placing the start or goal, PathPilot automatically returns to Obstacles
mode to prevent accidental endpoint changes.

## Project organization

```text
src/pathpilot/
├── environment.py       # UI-independent grid rules and state
├── game.py              # Pygame event loop and input coordination
├── planning/
│   ├── base.py          # Planner protocol and immutable search results
│   └── astar.py         # Deterministic four-direction A* planner
├── evaluation/
│   ├── scenarios.py     # Seeded maps and dynamic obstacle schedules
│   └── benchmark.py     # Headless paired benchmark runner
├── controllers/         # Visualization and navigation policies
├── metrics/             # Shared navigation measurements
└── ui/
    ├── layout.py        # Pygame-independent geometry and hit-testing
    └── pygame_renderer.py  # Visual rendering only
```

The core planner now finds shortest paths using Manhattan-distance A*. It
records exploration order, node count, path length, success, and planner-only
latency without modifying the environment. The UI animates explored cells
before revealing the final path. The Always-Replan baseline can also move the
agent automatically while deliberately discarding each route after one step.
Obstacle edits remain available during navigation and are incorporated by the
next planning call. Cached-Path navigation avoids that repeated work unless a
new obstacle intersects the route the agent still needs to follow.

## Run the tests

No third-party packages are needed to test the environment or planner:

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

## Run the benchmark

The default benchmark generates ten seeded scenarios—alternating static and
dynamic maps—and repeats both strategies three times on identical inputs:

```bash
PYTHONPATH=src .venv/bin/python -m pathpilot.evaluation.benchmark \
  --scenarios 10 \
  --repetitions 3 \
  --width 30 \
  --height 20 \
  --density 0.20 \
  --dynamic-obstacles 2 \
  --seed 42 \
  --output benchmarks
```

This writes `benchmarks/results.csv` and `benchmarks/summary.json`. Planning
latency includes only time spent inside A*. Compare timing results from the
same invocation because absolute latency varies by computer and system load.

The repository includes a [Phase 2 reference report](benchmarks/README.md)
generated from the default command and the accompanying raw paired results.
