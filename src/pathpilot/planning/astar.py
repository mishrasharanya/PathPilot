"""Deterministic A* pathfinding for four-connected PathPilot grids."""

from __future__ import annotations

import heapq
from itertools import count
from math import inf
from time import perf_counter_ns

from pathpilot.environment import GridEnvironment, Position
from pathpilot.planning.base import SearchResult


class AStarPlanner:
    """Find shortest grid paths using Manhattan-distance A*."""

    @staticmethod
    def heuristic(position: Position, goal: Position) -> int:
        """Estimate remaining four-direction movement using Manhattan distance."""
        return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

    def plan(
        self,
        environment: GridEnvironment,
        start: Position,
        goal: Position,
    ) -> SearchResult:
        """Return a shortest path without modifying the environment."""
        started_at = perf_counter_ns()
        if not environment.is_traversable(start) or not environment.is_traversable(goal):
            return self._failure((), started_at)

        sequence = count()
        frontier: list[tuple[int, int, int, Position]] = []
        start_heuristic = self.heuristic(start, goal)
        heapq.heappush(
            frontier,
            (start_heuristic, start_heuristic, next(sequence), start),
        )

        came_from: dict[Position, Position | None] = {start: None}
        cost_from_start: dict[Position, int] = {start: 0}
        explored: list[Position] = []
        closed: set[Position] = set()

        while frontier:
            _, _, _, current = heapq.heappop(frontier)
            if current in closed:
                continue

            closed.add(current)
            explored.append(current)
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                return SearchResult(
                    path=path,
                    explored_cells=tuple(explored),
                    nodes_explored=len(explored),
                    planning_latency_ns=perf_counter_ns() - started_at,
                )

            current_cost = cost_from_start[current]
            for neighbor in environment.neighbors(current):
                if neighbor in closed:
                    continue
                tentative_cost = current_cost + 1
                if tentative_cost >= cost_from_start.get(neighbor, inf):
                    continue

                cost_from_start[neighbor] = tentative_cost
                came_from[neighbor] = current
                estimated_remaining = self.heuristic(neighbor, goal)
                estimated_total = tentative_cost + estimated_remaining
                heapq.heappush(
                    frontier,
                    (
                        estimated_total,
                        estimated_remaining,
                        next(sequence),
                        neighbor,
                    ),
                )

        return self._failure(tuple(explored), started_at)

    @staticmethod
    def _reconstruct_path(
        came_from: dict[Position, Position | None],
        goal: Position,
    ) -> tuple[Position, ...]:
        path = [goal]
        current = came_from[goal]
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return tuple(path)

    @staticmethod
    def _failure(
        explored_cells: tuple[Position, ...],
        started_at: int,
    ) -> SearchResult:
        return SearchResult(
            path=(),
            explored_cells=explored_cells,
            nodes_explored=len(explored_cells),
            planning_latency_ns=perf_counter_ns() - started_at,
        )
