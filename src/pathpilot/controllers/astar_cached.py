"""Optimized A* controller that follows a valid cached route."""

from __future__ import annotations

from pathpilot.controllers.navigation import (
    NavigationState,
    TimedNavigationController,
    direction_between,
)
from pathpilot.environment import GridEnvironment, MoveResult, Position
from pathpilot.planning import PathPlanner


class CachedPathController(TimedNavigationController):
    """Reuse an A* route until an obstacle blocks its remaining cells."""

    def __init__(
        self,
        planner: PathPlanner | None = None,
        movement_interval_ms: int = 280,
    ) -> None:
        super().__init__(planner, movement_interval_ms)
        self._remaining_path: tuple[Position, ...] = ()

    @property
    def remaining_path(self) -> tuple[Position, ...]:
        return self._remaining_path

    def _reset_strategy(self) -> None:
        self._remaining_path = ()

    def _advance(self, environment: GridEnvironment) -> None:
        if self._needs_plan(environment):
            result = self.planner.plan(
                environment,
                environment.agent,
                environment.goal,
            )
            self.latest_result = result
            self.metrics.record_plan(result)
            self._remaining_path = result.path

            if not result.success:
                self.state = NavigationState.FAILED
                return

        if len(self._remaining_path) == 1:
            self.state = NavigationState.SUCCEEDED
            return

        direction = direction_between(
            self._remaining_path[0],
            self._remaining_path[1],
        )
        move_result = environment.move(direction)
        if move_result not in {MoveResult.MOVED, MoveResult.REACHED_GOAL}:
            self.state = NavigationState.FAILED
            return

        self._remaining_path = self._remaining_path[1:]
        self.metrics.record_movement()
        if environment.reached_goal:
            self.state = NavigationState.SUCCEEDED

    def _needs_plan(self, environment: GridEnvironment) -> bool:
        if len(self._remaining_path) < 2:
            return True
        if self._remaining_path[0] != environment.agent:
            return True
        return any(
            position in environment.obstacles
            for position in self._remaining_path[1:]
        )
