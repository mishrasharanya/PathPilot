import unittest

from pathpilot.controllers import CachedPathController, NavigationState
from pathpilot.environment import GridEnvironment


class CachedPathControllerTests(unittest.TestCase):
    def test_reuses_one_plan_for_an_unobstructed_route(self) -> None:
        environment = GridEnvironment(width=5, height=2, goal=(4, 0))
        controller = CachedPathController(movement_interval_ms=10)
        controller.start(environment)

        for _ in range(4):
            controller.update(10, environment)

        self.assertEqual(controller.state, NavigationState.SUCCEEDED)
        self.assertEqual(controller.metrics.planning_calls, 1)
        self.assertEqual(controller.metrics.replanning_count, 0)
        self.assertEqual(controller.metrics.path_length, 4)

    def test_replans_when_obstacle_blocks_remaining_path(self) -> None:
        environment = GridEnvironment(width=5, height=2, goal=(4, 0))
        controller = CachedPathController(movement_interval_ms=10)
        controller.start(environment)
        controller.update(10, environment)
        environment.add_obstacle((2, 0))

        controller.update(10, environment)

        self.assertEqual(controller.metrics.planning_calls, 2)
        self.assertEqual(controller.metrics.replanning_count, 1)
        self.assertNotIn((2, 0), controller.remaining_path)
        self.assertNotEqual(environment.agent, (2, 0))

    def test_does_not_replan_for_obstacle_off_remaining_path(self) -> None:
        environment = GridEnvironment(width=5, height=3, goal=(4, 0))
        controller = CachedPathController(movement_interval_ms=10)
        controller.start(environment)
        controller.update(10, environment)
        environment.add_obstacle((2, 2))

        controller.update(10, environment)

        self.assertEqual(controller.metrics.planning_calls, 1)
        self.assertEqual(controller.metrics.replanning_count, 0)

    def test_does_not_replan_for_obstacle_on_passed_cell(self) -> None:
        environment = GridEnvironment(width=5, height=2, goal=(4, 0))
        controller = CachedPathController(movement_interval_ms=10)
        controller.start(environment)
        controller.update(10, environment)
        controller.update(10, environment)
        environment.add_obstacle((0, 0))

        controller.update(10, environment)

        self.assertEqual(controller.metrics.planning_calls, 1)
        self.assertEqual(controller.metrics.replanning_count, 0)

    def test_reports_failure_when_replanned_route_is_unreachable(self) -> None:
        environment = GridEnvironment(width=4, height=2, goal=(3, 0))
        controller = CachedPathController(movement_interval_ms=10)
        controller.start(environment)
        controller.update(10, environment)
        environment.add_obstacle((2, 0))
        environment.add_obstacle((1, 1))

        controller.update(10, environment)

        self.assertEqual(controller.state, NavigationState.FAILED)
        self.assertEqual(controller.metrics.planning_calls, 2)

    def test_pause_and_stop_share_navigation_lifecycle(self) -> None:
        environment = GridEnvironment(width=4, height=2, goal=(3, 0))
        controller = CachedPathController(movement_interval_ms=10)
        controller.start(environment)
        self.assertTrue(controller.toggle_pause())
        controller.update(100, environment)
        self.assertEqual(environment.agent, environment.start)

        self.assertFalse(controller.toggle_pause())
        controller.update(10, environment)
        self.assertNotEqual(environment.agent, environment.start)

        controller.stop()
        self.assertEqual(controller.state, NavigationState.IDLE)
        self.assertEqual(controller.remaining_path, ())
        self.assertEqual(controller.metrics.planning_calls, 0)

    def test_cached_strategy_requires_fewer_plans_than_baseline(self) -> None:
        from pathpilot.controllers import AlwaysReplanController

        baseline_environment = GridEnvironment(width=6, height=2, goal=(5, 0))
        cached_environment = GridEnvironment(width=6, height=2, goal=(5, 0))
        baseline = AlwaysReplanController(movement_interval_ms=1)
        cached = CachedPathController(movement_interval_ms=1)
        baseline.start(baseline_environment)
        cached.start(cached_environment)

        for _ in range(5):
            baseline.update(1, baseline_environment)
            cached.update(1, cached_environment)

        self.assertEqual(baseline.state, NavigationState.SUCCEEDED)
        self.assertEqual(cached.state, NavigationState.SUCCEEDED)
        self.assertLess(
            cached.metrics.planning_calls,
            baseline.metrics.planning_calls,
        )
