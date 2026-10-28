"""Low-allocation observation encoding and inference for trained CNN policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import torch
from torch import Tensor

from pathpilot.environment import GridEnvironment, Position
from pathpilot.learning.actions import NavigationAction
from pathpilot.learning.checkpoints import load_checkpoint
from pathpilot.learning.encoding import (
    AGENT_CHANNEL,
    GOAL_CHANNEL,
    OBSTACLE_CHANNEL,
)


@dataclass(frozen=True, slots=True)
class ActionPrediction:
    """One model decision and the latency of its two processing stages."""

    action: NavigationAction
    encoding_latency_ns: int
    inference_latency_ns: int


class CachedObservationEncoder:
    """Reuse one tensor and update only environment cells that changed."""

    def __init__(self) -> None:
        self._tensor: Tensor | None = None
        self._width = 0
        self._height = 0
        self._obstacles: frozenset[Position] = frozenset()
        self._agent: Position | None = None
        self._goal: Position | None = None

    def encode(self, environment: GridEnvironment) -> tuple[Tensor, int]:
        """Return a `(1, 3, height, width)` tensor and encoding latency."""
        started_at = perf_counter_ns()
        if (
            self._tensor is None
            or self._width != environment.width
            or self._height != environment.height
        ):
            self._initialize(environment)
        else:
            self._update_obstacles(environment)
            self._update_single_cell_channel(
                AGENT_CHANNEL,
                previous=self._agent,
                current=environment.agent,
            )
            self._update_single_cell_channel(
                GOAL_CHANNEL,
                previous=self._goal,
                current=environment.goal,
            )
            self._agent = environment.agent
            self._goal = environment.goal
        if self._tensor is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("Observation tensor was not initialized")
        return self._tensor, perf_counter_ns() - started_at

    def _initialize(self, environment: GridEnvironment) -> None:
        self._width = environment.width
        self._height = environment.height
        self._tensor = torch.zeros(
            (1, 3, environment.height, environment.width),
            dtype=torch.float32,
        )
        for column, row in environment.obstacles:
            self._tensor[0, OBSTACLE_CHANNEL, row, column] = 1.0
        self._tensor[0, AGENT_CHANNEL, environment.agent[1], environment.agent[0]] = 1.0
        self._tensor[0, GOAL_CHANNEL, environment.goal[1], environment.goal[0]] = 1.0
        self._obstacles = frozenset(environment.obstacles)
        self._agent = environment.agent
        self._goal = environment.goal

    def _update_obstacles(self, environment: GridEnvironment) -> None:
        current = frozenset(environment.obstacles)
        for column, row in self._obstacles - current:
            self._tensor[0, OBSTACLE_CHANNEL, row, column] = 0.0  # type: ignore[index]
        for column, row in current - self._obstacles:
            self._tensor[0, OBSTACLE_CHANNEL, row, column] = 1.0  # type: ignore[index]
        self._obstacles = current

    def _update_single_cell_channel(
        self,
        channel: int,
        *,
        previous: Position | None,
        current: Position,
    ) -> None:
        if previous == current:
            return
        if previous is not None:
            self._tensor[0, channel, previous[1], previous[0]] = 0.0  # type: ignore[index]
        self._tensor[0, channel, current[1], current[0]] = 1.0  # type: ignore[index]


class TorchCNNPolicy:
    """Load one checkpoint once and produce timed navigation decisions."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        loaded = load_checkpoint(checkpoint_path)
        self.model = loaded.model
        self.dataset_manifest = loaded.dataset_manifest
        self.encoder = CachedObservationEncoder()

    def validate_environment(self, environment: GridEnvironment) -> None:
        """Reject grids unlike the model's training observations."""
        config = self.dataset_manifest.get("config", {})
        expected = (config.get("width"), config.get("height"))
        actual = (environment.width, environment.height)
        if expected != actual:
            raise ValueError(
                f"CNN was trained for a {expected[0]}x{expected[1]} grid, "
                f"not {actual[0]}x{actual[1]}"
            )

    def predict(self, environment: GridEnvironment) -> ActionPrediction:
        """Encode one state and select the highest-scoring action."""
        self.validate_environment(environment)
        observation, encoding_latency_ns = self.encoder.encode(environment)
        started_at = perf_counter_ns()
        action = self.model.predict_action(observation)
        inference_latency_ns = perf_counter_ns() - started_at
        return ActionPrediction(
            action=action,
            encoding_latency_ns=encoding_latency_ns,
            inference_latency_ns=inference_latency_ns,
        )
