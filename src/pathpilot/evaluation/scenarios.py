"""Deterministic benchmark scenarios for classical navigation policies."""

from __future__ import annotations

import random
from dataclasses import dataclass

from pathpilot.environment import GridEnvironment, Position


@dataclass(frozen=True, slots=True)
class ObstacleEvent:
    """Obstacle mutation applied before a numbered movement cycle."""

    movement_step: int
    position: Position
    blocked: bool = True

    def __post_init__(self) -> None:
        if self.movement_step < 0:
            raise ValueError("movement_step cannot be negative")


@dataclass(frozen=True, slots=True)
class BenchmarkScenario:
    """Complete immutable input for one paired navigation comparison."""

    name: str
    width: int
    height: int
    start: Position
    goal: Position
    obstacles: frozenset[Position]
    events: tuple[ObstacleEvent, ...] = ()

    @property
    def kind(self) -> str:
        return "dynamic" if self.events else "static"

    def create_environment(self) -> GridEnvironment:
        """Create a fresh mutable environment for one strategy run."""
        return GridEnvironment(
            self.width,
            self.height,
            start=self.start,
            goal=self.goal,
            obstacles=self.obstacles,
        )


def generate_scenario(
    *,
    seed: int,
    width: int = 30,
    height: int = 20,
    obstacle_density: float = 0.2,
    dynamic_obstacles: int = 0,
) -> BenchmarkScenario:
    """Generate a solvable seeded map with up to two route-blocking events."""
    if width < 6 or height < 6:
        raise ValueError("Benchmark grids must be at least 6 by 6")
    if not 0 <= obstacle_density < 1:
        raise ValueError("obstacle_density must be in [0, 1)")
    if not 0 <= dynamic_obstacles <= 2:
        raise ValueError("dynamic_obstacles must be between 0 and 2")

    start = (0, 0)
    goal = (width - 1, height - 1)
    candidates = [
        (column, row)
        for row in range(height)
        for column in range(width)
        if (column, row) not in {start, goal}
    ]
    generator = random.Random(seed)
    generator.shuffle(candidates)
    count = int(len(candidates) * obstacle_density)
    obstacles = set(candidates[:count])

    # Parallel corridors guarantee an initial path and detours around the two
    # deterministic dynamic blockers.
    guaranteed_cells = {
        (column, row)
        for column in range(width)
        for row in (0, 1)
    }
    guaranteed_cells.update(
        {
            (column, row)
            for column in (width - 2, width - 1)
            for row in range(height)
        }
    )
    obstacles.difference_update(guaranteed_cells)

    events: list[ObstacleEvent] = []
    if dynamic_obstacles >= 1:
        events.append(
            ObstacleEvent(
                movement_step=1,
                position=(max(3, width // 3), 0),
            )
        )
    if dynamic_obstacles >= 2:
        events.append(
            ObstacleEvent(
                movement_step=4,
                position=(width - 1, height // 2),
            )
        )

    return BenchmarkScenario(
        name=f"seed-{seed}-{('dynamic' if events else 'static')}",
        width=width,
        height=height,
        start=start,
        goal=goal,
        obstacles=frozenset(obstacles),
        events=tuple(events),
    )


def generate_suite(
    *,
    scenario_count: int,
    seed: int,
    width: int,
    height: int,
    obstacle_density: float,
    dynamic_obstacles: int,
) -> tuple[BenchmarkScenario, ...]:
    """Generate alternating static and dynamic scenarios from stable seeds."""
    if scenario_count <= 0:
        raise ValueError("scenario_count must be positive")
    return tuple(
        generate_scenario(
            seed=seed + index,
            width=width,
            height=height,
            obstacle_density=obstacle_density,
            dynamic_obstacles=(dynamic_obstacles if index % 2 else 0),
        )
        for index in range(scenario_count)
    )
