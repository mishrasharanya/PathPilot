"""Timed autonomous navigation driven only by learned CNN predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pathpilot.controllers.navigation import NavigationState
from pathpilot.environment import GridEnvironment, MoveResult, Position
from pathpilot.learning.actions import NavigationAction


class Prediction(Protocol):
    action: NavigationAction
    encoding_latency_ns: int
    inference_latency_ns: int


class LearnedPolicy(Protocol):
    def validate_environment(self, environment: GridEnvironment) -> None: ...
    def predict(self, environment: GridEnvironment) -> Prediction: ...


@dataclass(slots=True)
class CNNNavigationMetrics:
    """Measurements specific to learned, single-step decisions."""

    inference_calls: int = 0
    encoding_latency_ns: int = 0
    inference_latency_ns: int = 0
    path_length: int = 0
    invalid_actions: int = 0
    loop_detections: int = 0

    @property
    def encoding_latency_ms(self) -> float:
        return self.encoding_latency_ns / 1_000_000

    @property
    def inference_latency_ms(self) -> float:
        return self.inference_latency_ns / 1_000_000

    @property
    def total_decision_latency_ms(self) -> float:
        return (self.encoding_latency_ns + self.inference_latency_ns) / 1_000_000


class CNNNavigationController:
    """Move with CNN actions until success, invalid-loop, or step limit."""

    def __init__(
        self,
        policy: LearnedPolicy,
        *,
        movement_interval_ms: int = 280,
        max_steps: int = 600,
        repeated_decision_limit: int = 3,
    ) -> None:
        if movement_interval_ms <= 0:
            raise ValueError("movement_interval_ms must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if repeated_decision_limit <= 0:
            raise ValueError("repeated_decision_limit must be positive")
        self.policy = policy
        self.movement_interval_ms = movement_interval_ms
        self.max_steps = max_steps
        self.repeated_decision_limit = repeated_decision_limit
        self.state = NavigationState.IDLE
        self.metrics = CNNNavigationMetrics()
        self.latest_result = None
        self._elapsed_ms = 0
        self._decision_counts: dict[tuple[Position, NavigationAction], int] = {}

    @property
    def is_active(self) -> bool:
        return self.state in {NavigationState.RUNNING, NavigationState.PAUSED}

    def start(self, environment: GridEnvironment) -> None:
        self.policy.validate_environment(environment)
        self.metrics = CNNNavigationMetrics()
        self._elapsed_ms = 0
        self._decision_counts.clear()
        self.state = (
            NavigationState.SUCCEEDED
            if environment.reached_goal
            else NavigationState.RUNNING
        )

    def update(self, elapsed_ms: int, environment: GridEnvironment) -> None:
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
        if self.state is NavigationState.RUNNING:
            self.state = NavigationState.PAUSED
            return True
        if self.state is NavigationState.PAUSED:
            self.state = NavigationState.RUNNING
        return False

    def stop(self) -> None:
        self.state = NavigationState.IDLE
        self.metrics = CNNNavigationMetrics()
        self._elapsed_ms = 0
        self._decision_counts.clear()

    def _advance(self, environment: GridEnvironment) -> None:
        if self.metrics.inference_calls >= self.max_steps:
            self.state = NavigationState.FAILED
            return

        prediction = self.policy.predict(environment)
        self.metrics.inference_calls += 1
        self.metrics.encoding_latency_ns += prediction.encoding_latency_ns
        self.metrics.inference_latency_ns += prediction.inference_latency_ns
        decision = (environment.agent, prediction.action)
        decision_count = self._decision_counts.get(decision, 0) + 1
        self._decision_counts[decision] = decision_count
        if decision_count > self.repeated_decision_limit:
            self.metrics.loop_detections += 1
            self.state = NavigationState.FAILED
            return

        result = environment.move(prediction.action.direction)
        if result not in {MoveResult.MOVED, MoveResult.REACHED_GOAL}:
            self.metrics.invalid_actions += 1
            if decision_count >= self.repeated_decision_limit:
                self.metrics.loop_detections += 1
                self.state = NavigationState.FAILED
            return

        self.metrics.path_length += 1
        if environment.reached_goal:
            self.state = NavigationState.SUCCEEDED
