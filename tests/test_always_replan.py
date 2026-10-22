import unittest

from pathpilot.controllers import AlwaysReplanController, NavigationState
from pathpilot.environment import GridEnvironment


class AlwaysReplanControllerTests(unittest.TestCase):
    def test_replans_before_every_movement(self) -> None:
        environment = GridEnvironment(width=4, height=2, goal=(3, 0))
        controller = AlwaysReplanController(movement_interval_ms=10)
        controller.start(environment)

        controller.update(10, environment)
        controller.update(10, environment)

        self.assertEqual(environment.agent, (2, 0))
        self.assertEqual(controller.metrics.planning_calls, 2)
        self.assertEqual(controller.metrics.replanning_count, 1)
        self.assertEqual(controller.metrics.path_length, 2)

    def test_waits_for_movement_interval(self) -> None:
        environment = GridEnvironment(width=3, height=2)
        controller = AlwaysReplanController(movement_interval_ms=10)
        controller.start(environment)

        controller.update(9, environment)

        self.assertEqual(environment.agent, environment.start)
        self.assertEqual(controller.metrics.planning_calls, 0)

    def test_reaches_goal_and_preserves_metrics(self) -> None:
        environment = GridEnvironment(width=2, height=1)
        controller = AlwaysReplanController(movement_interval_ms=10)
        controller.start(environment)

        controller.update(10, environment)

        self.assertEqual(controller.state, NavigationState.SUCCEEDED)
        self.assertTrue(environment.reached_goal)
        self.assertEqual(controller.metrics.planning_calls, 1)
        self.assertEqual(controller.metrics.path_length, 1)

    def test_reports_no_path(self) -> None:
        environment = GridEnvironment(
            width=3,
            height=3,
            obstacles={(1, 0), (0, 1)},
        )
        controller = AlwaysReplanController(movement_interval_ms=10)
        controller.start(environment)

        controller.update(10, environment)

        self.assertEqual(controller.state, NavigationState.FAILED)
        self.assertEqual(controller.metrics.planning_calls, 1)
        self.assertEqual(controller.metrics.path_length, 0)

    def test_new_obstacle_is_seen_by_next_replan(self) -> None:
        environment = GridEnvironment(width=5, height=2, goal=(4, 0))
        controller = AlwaysReplanController(movement_interval_ms=10)
        controller.start(environment)
        controller.update(10, environment)
        environment.add_obstacle((2, 0))

        controller.update(10, environment)

        self.assertEqual(controller.metrics.planning_calls, 2)
        self.assertNotEqual(environment.agent, (2, 0))
        self.assertNotIn((2, 0), controller.latest_result.path)

    def test_pause_and_resume(self) -> None:
        environment = GridEnvironment(width=3, height=2)
        controller = AlwaysReplanController(movement_interval_ms=10)
        controller.start(environment)
        self.assertTrue(controller.toggle_pause())

        controller.update(100, environment)
        self.assertEqual(environment.agent, environment.start)

        self.assertFalse(controller.toggle_pause())
        controller.update(10, environment)
        self.assertNotEqual(environment.agent, environment.start)

    def test_stop_clears_run_data(self) -> None:
        environment = GridEnvironment(width=3, height=2)
        controller = AlwaysReplanController(movement_interval_ms=10)
        controller.start(environment)
        controller.update(10, environment)

        controller.stop()

        self.assertEqual(controller.state, NavigationState.IDLE)
        self.assertIsNone(controller.latest_result)
        self.assertEqual(controller.metrics.planning_calls, 0)

    def test_rejects_invalid_timing(self) -> None:
        with self.assertRaises(ValueError):
            AlwaysReplanController(movement_interval_ms=0)
        controller = AlwaysReplanController()
        with self.assertRaises(ValueError):
            controller.update(-1, GridEnvironment())
