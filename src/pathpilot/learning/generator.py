"""Generate reproducible navigation datasets labeled by A*."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from pathpilot.environment import GridEnvironment, Position
from pathpilot.learning.actions import NavigationAction
from pathpilot.learning.dataset import NavigationDataset
from pathpilot.learning.encoding import encode_environment
from pathpilot.learning.samples import NavigationSample
from pathpilot.planning.astar import AStarPlanner


@dataclass(frozen=True, slots=True)
class DatasetGenerationConfig:
    """Complete reproducible configuration for one generation run."""

    maze_count: int = 100
    width: int = 20
    height: int = 15
    obstacle_density: float = 0.2
    seed: int = 42
    train_fraction: float = 0.8
    validation_fraction: float = 0.1

    def validate(self) -> None:
        if self.maze_count < 3:
            raise ValueError("maze_count must be at least 3")
        if self.width < 2 or self.height < 2:
            raise ValueError("Grid dimensions must be at least 2 by 2")
        if not 0 <= self.obstacle_density < 1:
            raise ValueError("obstacle_density must be in [0, 1)")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("Train and validation fractions must leave a test split")


@dataclass(frozen=True, slots=True)
class LabeledMaze:
    """All teacher samples produced from one solved maze."""

    maze_id: str
    samples: tuple[NavigationSample, ...]


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """Train, validation, and test sets split at maze granularity."""

    train: NavigationDataset
    validation: NavigationDataset
    test: NavigationDataset


def generate_labeled_mazes(
    config: DatasetGenerationConfig,
) -> tuple[LabeledMaze, ...]:
    """Generate exactly the requested number of solvable labeled mazes."""
    config.validate()
    generator = random.Random(config.seed)
    planner = AStarPlanner()
    labeled_mazes: list[LabeledMaze] = []
    attempts = 0
    maximum_attempts = config.maze_count * 100

    while len(labeled_mazes) < config.maze_count and attempts < maximum_attempts:
        maze_seed = generator.randrange(0, 2**63)
        maze_id = f"maze-{len(labeled_mazes):06d}-seed-{maze_seed}"
        environment = _create_random_environment(config, seed=maze_seed)
        result = planner.plan(environment, environment.start, environment.goal)
        attempts += 1
        if not result.success:
            continue

        samples = _label_path(environment, result.path, maze_id=maze_id)
        labeled_mazes.append(LabeledMaze(maze_id=maze_id, samples=samples))

    if len(labeled_mazes) != config.maze_count:
        raise RuntimeError(
            f"Generated only {len(labeled_mazes)} solvable mazes after "
            f"{maximum_attempts} attempts"
        )
    return tuple(labeled_mazes)


def split_labeled_mazes(
    labeled_mazes: Sequence[LabeledMaze],
    config: DatasetGenerationConfig,
) -> DatasetSplits:
    """Split whole mazes, preventing path-step leakage between datasets."""
    config.validate()
    if len(labeled_mazes) != config.maze_count:
        raise ValueError("labeled_mazes count must match the generation config")

    shuffled = list(labeled_mazes)
    random.Random(config.seed + 1).shuffle(shuffled)
    train_count = int(config.maze_count * config.train_fraction)
    validation_count = int(config.maze_count * config.validation_fraction)
    if train_count == 0 or validation_count == 0:
        raise ValueError("Configuration produces an empty train or validation split")
    if train_count + validation_count >= config.maze_count:
        raise ValueError("Configuration produces an empty test split")

    train_mazes = shuffled[:train_count]
    validation_mazes = shuffled[train_count : train_count + validation_count]
    test_mazes = shuffled[train_count + validation_count :]
    return DatasetSplits(
        train=_make_dataset(train_mazes, config),
        validation=_make_dataset(validation_mazes, config),
        test=_make_dataset(test_mazes, config),
    )


def generate_datasets(config: DatasetGenerationConfig) -> DatasetSplits:
    """Generate solved mazes and return leakage-safe dataset splits."""
    return split_labeled_mazes(generate_labeled_mazes(config), config)


def write_datasets(
    splits: DatasetSplits,
    config: DatasetGenerationConfig,
    output_directory: Path,
) -> None:
    """Save all splits and a human-readable reproducibility manifest."""
    output_directory.mkdir(parents=True, exist_ok=True)
    split_map = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    for name, dataset in split_map.items():
        dataset.save(output_directory / f"{name}.json.gz")

    manifest = {
        "generator": "astar",
        "config": asdict(config),
        "splits": {
            name: {
                "maze_count": len({sample.maze_id for sample in dataset.samples}),
                "sample_count": len(dataset.samples),
            }
            for name, dataset in split_map.items()
        },
    }
    with (output_directory / "manifest.json").open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")


def _create_random_environment(
    config: DatasetGenerationConfig,
    *,
    seed: int,
) -> GridEnvironment:
    all_positions: list[Position] = [
        (column, row)
        for row in range(config.height)
        for column in range(config.width)
    ]
    generator = random.Random(seed)
    start = generator.choice(all_positions)
    minimum_distance = max(2, (config.width + config.height) // 3)
    possible_goals = [
        position
        for position in all_positions
        if _manhattan_distance(start, position) >= minimum_distance
    ]
    goal = generator.choice(possible_goals)
    candidates = [
        position for position in all_positions if position not in {start, goal}
    ]
    generator.shuffle(candidates)
    obstacle_count = int(len(candidates) * config.obstacle_density)
    return GridEnvironment(
        width=config.width,
        height=config.height,
        start=start,
        goal=goal,
        obstacles=candidates[:obstacle_count],
    )


def _manhattan_distance(first: Position, second: Position) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def _label_path(
    environment: GridEnvironment,
    path: Sequence[Position],
    *,
    maze_id: str,
) -> tuple[NavigationSample, ...]:
    samples: list[NavigationSample] = []
    for step_index, (current, next_position) in enumerate(zip(path, path[1:])):
        if environment.agent != current:
            raise ValueError("Environment agent and teacher path are out of sync")
        action = NavigationAction.from_transition(current, next_position)
        samples.append(
            NavigationSample(
                maze_id=maze_id,
                step_index=step_index,
                observation=encode_environment(environment),
                action=action,
            )
        )
        environment.move(action.direction)
    return tuple(samples)


def _make_dataset(
    mazes: Sequence[LabeledMaze],
    config: DatasetGenerationConfig,
) -> NavigationDataset:
    return NavigationDataset.create(
        width=config.width,
        height=config.height,
        samples=(sample for maze in mazes for sample in maze.samples),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mazes", type=int, default=100)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("datasets/default"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = DatasetGenerationConfig(
        maze_count=args.mazes,
        width=args.width,
        height=args.height,
        obstacle_density=args.density,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    splits = generate_datasets(config)
    write_datasets(splits, config, args.output)
    print(f"Datasets written to {args.output.resolve()}")
    for name, dataset in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        maze_count = len({sample.maze_id for sample in dataset.samples})
        print(f"{name}: {maze_count} mazes, {len(dataset.samples)} samples")


if __name__ == "__main__":
    main()
