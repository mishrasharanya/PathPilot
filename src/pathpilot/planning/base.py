"""Shared contracts for PathPilot path-planning algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pathpilot.environment import GridEnvironment, Position


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Immutable output produced by a path-planning operation.

    A successful path includes both its requested start and goal. An empty path
    represents failure. ``explored_cells`` preserves search order so the UI can
    animate exploration later without coupling the planner to Pygame.
    """

    path: tuple[Position, ...]
    explored_cells: tuple[Position, ...]
    nodes_explored: int
    planning_latency_ns: int

    def __post_init__(self) -> None:
        if self.nodes_explored < 0:
            raise ValueError("nodes_explored cannot be negative")
        if self.planning_latency_ns < 0:
            raise ValueError("planning_latency_ns cannot be negative")
        if self.nodes_explored != len(self.explored_cells):
            raise ValueError(
                "nodes_explored must equal the number of explored cells"
            )

    @property
    def success(self) -> bool:
        """Whether the planner found a path."""
        return bool(self.path)

    @property
    def path_length(self) -> int:
        """Number of movements represented by the path."""
        return max(0, len(self.path) - 1)


@runtime_checkable
class PathPlanner(Protocol):
    """Interface implemented by every classical path planner."""

    def plan(
        self,
        environment: GridEnvironment,
        start: Position,
        goal: Position,
    ) -> SearchResult:
        """Find a path without mutating the supplied environment."""
        ...
