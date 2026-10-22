"""UI-independent grid state and navigation rules for PathPilot."""

from __future__ import annotations

from enum import Enum
from typing import Iterable

Position = tuple[int, int]


class Direction(Enum):
    """One-cell movement directions in grid coordinates."""

    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class MoveResult(Enum):
    """Outcome of a requested agent movement."""

    MOVED = "moved"
    REACHED_GOAL = "reached_goal"
    BLOCKED_BY_OBSTACLE = "blocked_by_obstacle"
    OUT_OF_BOUNDS = "out_of_bounds"


class GridEnvironment:
    """Mutable rectangular grid containing an agent, endpoints, and obstacles.

    The environment deliberately has no dependency on Pygame. User interfaces,
    algorithms, and future learning agents all interact through the same public
    methods and therefore receive identical validation behavior.
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 15,
        *,
        start: Position = (0, 0),
        goal: Position | None = None,
        obstacles: Iterable[Position] = (),
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Grid dimensions must be positive")

        self.width = width
        self.height = height
        self._default_start = start
        self._default_goal = goal if goal is not None else (width - 1, height - 1)
        self.start = self._default_start
        self.goal = self._default_goal
        self.agent = self.start
        self.obstacles = set(obstacles)
        self._validate_initial_state()

    def _validate_initial_state(self) -> None:
        if not self.in_bounds(self.start):
            raise ValueError(f"Start position {self.start} is outside the grid")
        if not self.in_bounds(self.goal):
            raise ValueError(f"Goal position {self.goal} is outside the grid")
        if self.start == self.goal:
            raise ValueError("Start and goal positions must be different")
        invalid = {position for position in self.obstacles if not self.in_bounds(position)}
        if invalid:
            raise ValueError(f"Obstacles outside the grid: {sorted(invalid)}")
        if self.start in self.obstacles or self.goal in self.obstacles:
            raise ValueError("Obstacles cannot cover the start or goal")

    def in_bounds(self, position: Position) -> bool:
        column, row = position
        return 0 <= column < self.width and 0 <= row < self.height

    @property
    def reached_goal(self) -> bool:
        return self.agent == self.goal

    def move(self, direction: Direction) -> MoveResult:
        if not isinstance(direction, Direction):
            raise TypeError("direction must be a Direction")

        delta_column, delta_row = direction.value
        destination = (
            self.agent[0] + delta_column,
            self.agent[1] + delta_row,
        )
        if not self.in_bounds(destination):
            return MoveResult.OUT_OF_BOUNDS
        if destination in self.obstacles:
            return MoveResult.BLOCKED_BY_OBSTACLE

        self.agent = destination
        if self.reached_goal:
            return MoveResult.REACHED_GOAL
        return MoveResult.MOVED

    def add_obstacle(self, position: Position) -> bool:
        """Add an obstacle if the cell is editable; return whether it changed."""
        if not self.in_bounds(position):
            return False
        if position in {self.start, self.goal, self.agent}:
            return False
        before = len(self.obstacles)
        self.obstacles.add(position)
        return len(self.obstacles) > before

    def remove_obstacle(self, position: Position) -> bool:
        """Remove an obstacle; return whether one existed in that cell."""
        if position not in self.obstacles:
            return False
        self.obstacles.remove(position)
        return True

    def set_start(self, position: Position) -> bool:
        """Place the start on a valid cell and return the agent to it."""
        if not self.in_bounds(position):
            return False
        if position in self.obstacles or position == self.goal:
            return False
        self.start = position
        self.agent = position
        return True

    def set_goal(self, position: Position) -> bool:
        """Place the goal on a valid non-start cell."""
        if not self.in_bounds(position):
            return False
        if position in self.obstacles or position == self.start:
            return False
        self.goal = position
        return True

    def reset_agent(self) -> None:
        """Return the agent to the current start without changing the board."""
        self.agent = self.start

    def reset(self) -> None:
        """Clear obstacles and restore the environment's original endpoints."""
        self.start = self._default_start
        self.goal = self._default_goal
        self.agent = self.start
        self.obstacles.clear()
