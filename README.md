# PathPilot

PathPilot is an interactive navigation game that will grow incrementally from a
manual 2D grid into a complete AI navigation and evaluation project.

## Current status

**Phase 3 — Milestone 5:** learned CNN navigation and Pygame integration.

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
- Stable four-action labels compatible with the existing movement rules
- Immutable three-channel observations for obstacles, agent, and goal
- Validated labeled samples and versioned compressed dataset storage
- Seeded A*-teacher generation with maze-level train/validation/test splits
- Compact PyTorch CNN and lazy tensor dataset adapter
- Explicit training/validation loops, history, and best-model checkpoints
- Cached-tensor CNN inference with separate latency measurements
- Interactive learned navigation with collision, loop, and step safeguards

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
| CNN | Navigate with the trained model and no hidden A* fallback |
| Pause / Resume | Pause or continue visualization or autonomous navigation |
| Stop | Stop the active AI mode and clear its run state |

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
│   └── cnn_navigation.py # Learned-policy lifecycle and safeguards
├── metrics/             # Shared navigation measurements
├── learning/
│   ├── actions.py       # Stable model labels and movement conversion
│   ├── encoding.py      # Three-channel environment observations
│   ├── samples.py       # One validated A*-labeled training example
│   ├── dataset.py       # Versioned dataset validation and storage
│   ├── generator.py     # Seeded maze generation and A* teacher labels
│   ├── torch_dataset.py # PyTorch tensor and DataLoader adapter
│   ├── model.py         # Compact convolutional navigation policy
│   ├── training.py      # Forward/backward passes and epoch metrics
│   ├── checkpoints.py   # Versioned safe model persistence
│   ├── train_cli.py     # Reproducible training command
│   └── inference.py     # Cached tensor updates and timed predictions
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

Phase 3 begins without coupling the environment to a machine-learning library.
An observation is an immutable `(3, height, width)` structure containing one
binary channel each for obstacles, the current agent, and the goal. Labeled
samples pair that observation with one of `UP`, `DOWN`, `LEFT`, or `RIGHT`.
Datasets are stored as versioned gzip-compressed JSON. Randomized, well-separated
start and goal positions reduce directional label bias. A seeded generator
rejects unsolvable maps, labels every successful route with A*, and splits whole
mazes so closely related path steps cannot leak between datasets.

## Generate supervised-learning data

Generate 100 reproducible mazes and write the three splits plus a manifest:

```bash
PYTHONPATH=src python3 -m pathpilot.learning.generator \
  --mazes 100 \
  --width 20 \
  --height 15 \
  --density 0.20 \
  --seed 42 \
  --output datasets/default
```

The output directory contains `train.json.gz`, `validation.json.gz`,
`test.json.gz`, and `manifest.json`. Generated datasets are ignored by Git;
the command and manifest settings make them reproducible.

## Run the tests

Install the optional ML dependencies to run the complete test suite:

```bash
python3 -m pip install -e ".[ml]"
```

Then run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The environment, planner, and Pygame game remain independent of PyTorch. Only
the Phase 3 model, tensor adapter, and their tests require the ML dependencies.

## Supervised CNN

`PathPilotCNN` uses two small convolution layers (16 then 32 channels), adaptive
pooling, a 64-unit hidden layer, and four raw action logits. It contains fewer
than 50,000 trainable parameters. Interactive prediction uses inference mode
and selects the largest logit directly, without computing softmax. Training
records classification accuracy and duration; end-to-end navigation success
and inference latency will be measured in the evaluation milestone.

## Train the supervised CNN

After generating `datasets/default`, run:

```bash
PYTHONPATH=src .venv/bin/python -m pathpilot.learning.train_cli \
  --dataset datasets/default \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --seed 42 \
  --device cpu \
  --output artifacts/models/cnn-v1
```

The command writes `best_model.pt`, `history.json`, and
`training_config.json`. Generated model artifacts are ignored by Git because
the dataset seed and saved configuration reproduce the experiment. The first
local CPU run reached 63.7% best validation action accuracy at epoch 11. This
is a classification baseline, not yet a navigation success-rate measurement;
the held-out test split remains untouched for the evaluation milestone.

The CNN controller loads `artifacts/models/cnn-v1/best_model.pt` once. During
navigation it reuses one observation tensor and updates only cells affected by
agent, goal, or obstacle changes. The UI reports encoding and model-inference
latency separately. Invalid actions, repeated decisions, and a maximum step
count terminate failed runs safely. If the ML dependencies or checkpoint are
missing, the CNN control explains the issue without disabling manual or A*
navigation.

## Run the interface

Install the project and launch the manual and A* modes:

```bash
python -m pip install -e .
pathpilot
```

To use the CNN button, install the ML extras and train the checkpoint first:

```bash
python -m pip install -e ".[ml]"
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
