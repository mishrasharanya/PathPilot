"""Baseline A* controller that discards its route after every movement."""

from __future__ import annotations

from pathpilot.controllers.navigation import (
    NavigationState,
    TimedNavigationController,
    direction_between,
)
from pathpilot.environment import GridEnvironment, MoveResult


class AlwaysReplanController(TimedNavigationController):
    """Run A* before every single agent movement.

    This intentionally inefficient policy is the Phase 2 baseline against
    which cached-path navigation is compared.
    """

    def _advance(self, environment: GridEnvironment) -> None:
        result = self.planner.plan(
            environment,
            environment.agent,
            environment.goal,
        )
        self.latest_result = result
        self.metrics.record_plan(result)

        if not result.success:
            self.state = NavigationState.FAILED
            return
        if len(result.path) == 1:
            self.state = NavigationState.SUCCEEDED
            return

        direction = direction_between(result.path[0], result.path[1])
        move_result = environment.move(direction)
        if move_result not in {MoveResult.MOVED, MoveResult.REACHED_GOAL}:
            self.state = NavigationState.FAILED
            return
        self.metrics.record_movement()
        if environment.reached_goal:
            self.state = NavigationState.SUCCEEDED
