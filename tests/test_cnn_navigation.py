import unittest
from dataclasses import dataclass

from pathpilot.controllers import CNNNavigationController, NavigationState
from pathpilot.environment import GridEnvironment
from pathpilot.learning.actions import NavigationAction


@dataclass(frozen=True)
class FakePrediction:
    action: NavigationAction
    encoding_latency_ns: int = 100
    inference_latency_ns: int = 200


class FakePolicy:
    def __init__(self, actions: list[NavigationAction]) -> None:
        self.actions = actions
        self.calls = 0

    def validate_environment(self, environment: GridEnvironment) -> None:
        return None

    def predict(self, environment: GridEnvironment) -> FakePrediction:
        action = self.actions[min(self.calls, len(self.actions) - 1)]
        self.calls += 1
        return FakePrediction(action)


class CNNNavigationControllerTests(unittest.TestCase):
    def test_predictions_move_agent_to_goal_and_record_latency(self) -> None:
        environment = GridEnvironment(width=3, height=2, goal=(2, 0))
        controller = CNNNavigationController(
            FakePolicy([NavigationAction.RIGHT]),
            movement_interval_ms=1,
        )
        controller.start(environment)
        controller.update(1, environment)
        controller.update(1, environment)

        self.assertEqual(controller.state, NavigationState.SUCCEEDED)
        self.assertEqual(controller.metrics.inference_calls, 2)
        self.assertEqual(controller.metrics.path_length, 2)
        self.assertEqual(controller.metrics.encoding_latency_ns, 200)
        self.assertEqual(controller.metrics.inference_latency_ns, 400)

    def test_repeated_invalid_action_terminates_run(self) -> None:
        environment = GridEnvironment(width=3, height=2)
        controller = CNNNavigationController(
            FakePolicy([NavigationAction.UP]),
            movement_interval_ms=1,
            repeated_decision_limit=2,
        )
        controller.start(environment)
        controller.update(1, environment)
        controller.update(1, environment)

        self.assertEqual(controller.state, NavigationState.FAILED)
        self.assertEqual(controller.metrics.invalid_actions, 2)
        self.assertEqual(controller.metrics.loop_detections, 1)

    def test_repeated_state_action_loop_is_detected(self) -> None:
        environment = GridEnvironment(width=3, height=2)
        controller = CNNNavigationController(
            FakePolicy(
                [
                    NavigationAction.RIGHT,
                    NavigationAction.LEFT,
                    NavigationAction.RIGHT,
                ]
            ),
            movement_interval_ms=1,
            repeated_decision_limit=1,
        )
        controller.start(environment)
        for _ in range(3):
            controller.update(1, environment)
        self.assertEqual(controller.state, NavigationState.FAILED)
        self.assertEqual(controller.metrics.loop_detections, 1)

    def test_step_limit_stops_navigation(self) -> None:
        environment = GridEnvironment(width=4, height=2)
        controller = CNNNavigationController(
            FakePolicy([NavigationAction.RIGHT, NavigationAction.LEFT]),
            movement_interval_ms=1,
            max_steps=2,
        )
        controller.start(environment)
        for _ in range(3):
            controller.update(1, environment)
        self.assertEqual(controller.state, NavigationState.FAILED)
        self.assertEqual(controller.metrics.inference_calls, 2)

    def test_pause_and_stop_follow_shared_lifecycle(self) -> None:
        environment = GridEnvironment(width=3, height=2)
        controller = CNNNavigationController(
            FakePolicy([NavigationAction.RIGHT]),
            movement_interval_ms=1,
        )
        controller.start(environment)
        self.assertTrue(controller.toggle_pause())
        controller.update(1, environment)
        self.assertEqual(environment.agent, environment.start)
        controller.stop()
        self.assertEqual(controller.state, NavigationState.IDLE)
        self.assertEqual(controller.metrics.inference_calls, 0)
