"""Shared lifecycle for timed autonomous navigation controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from pathpilot.environment import Direction, GridEnvironment, Position
from pathpilot.metrics import NavigationMetrics
from pathpilot.planning import AStarPlanner, PathPlanner, SearchResult


class NavigationState(Enum):
    """Lifecycle state shared by autonomous navigation runs."""

    IDLE = "Idle"
    RUNNING = "Running"
    PAUSED = "Paused"
    SUCCEEDED = "Succeeded"
    FAILED = "No path"


class TimedNavigationController(ABC):
    """Common timing, metrics, and lifecycle behavior for navigation policies."""

    def __init__(
        self,
        planner: PathPlanner | None = None,
        movement_interval_ms: int = 280,
    ) -> None:
        if movement_interval_ms <= 0:
            raise ValueError("movement_interval_ms must be positive")
        self.planner = planner or AStarPlanner()
        self.movement_interval_ms = movement_interval_ms
        self.state = NavigationState.IDLE
        self.metrics = NavigationMetrics()
        self.latest_result: SearchResult | None = None
        self._elapsed_ms = 0

    @property
    def is_active(self) -> bool:
        return self.state in {NavigationState.RUNNING, NavigationState.PAUSED}

    def start(self, environment: GridEnvironment) -> None:
        """Start a fresh run from the agent's current position."""
        self.metrics = NavigationMetrics()
        self.latest_result = None
        self._elapsed_ms = 0
        self._reset_strategy()
        self.state = (
            NavigationState.SUCCEEDED
            if environment.reached_goal
            else NavigationState.RUNNING
        )

    def update(self, elapsed_ms: int, environment: GridEnvironment) -> None:
        """Advance the policy once when its movement interval elapses."""
        if elapsed_ms < 0:
            raise ValueError("elapsed_ms cannot be negative")
        if self.state is not NavigationState.RUNNING:
            return

        self._elapsed_ms += elapsed_ms
        if self._elapsed_ms < self.movement_interval_ms:
            return
        self._elapsed_ms %= self.movement_interval_ms
        self._advance(environment)

    def toggle_pause(self) -> bool:
        """Toggle a live run and return whether it is now paused."""
        if self.state is NavigationState.RUNNING:
            self.state = NavigationState.PAUSED
            return True
        if self.state is NavigationState.PAUSED:
            self.state = NavigationState.RUNNING
        return False

    def stop(self) -> None:
        """Stop the run and clear its result and metrics."""
        self.state = NavigationState.IDLE
        self.metrics = NavigationMetrics()
        self.latest_result = None
        self._elapsed_ms = 0
        self._reset_strategy()

    @abstractmethod
    def _advance(self, environment: GridEnvironment) -> None:
        """Perform one policy-specific planning and movement cycle."""

    def _reset_strategy(self) -> None:
        """Clear policy-specific state when starting or stopping."""


def direction_between(current: Position, following: Position) -> Direction:
    """Return the direction between two adjacent cells."""
    delta = (following[0] - current[0], following[1] - current[1])
    for direction in Direction:
        if direction.value == delta:
            return direction
    raise ValueError("Planner returned a path containing non-adjacent cells")
