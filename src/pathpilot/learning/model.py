"""Compact convolutional policy for supervised PathPilot navigation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pathpilot.learning.actions import NavigationAction
from pathpilot.learning.encoding import CHANNEL_COUNT


@dataclass(frozen=True, slots=True)
class CNNConfig:
    """Architecture settings kept explicit for reproducible experiments."""

    input_channels: int = CHANNEL_COUNT
    first_conv_channels: int = 16
    second_conv_channels: int = 32
    pooled_height: int = 3
    pooled_width: int = 4
    hidden_features: int = 64
    action_count: int = len(NavigationAction)

    def validate(self) -> None:
        values = (
            self.input_channels,
            self.first_conv_channels,
            self.second_conv_channels,
            self.pooled_height,
            self.pooled_width,
            self.hidden_features,
            self.action_count,
        )
        if any(value <= 0 for value in values):
            raise ValueError("All CNN configuration values must be positive")
        if self.input_channels != CHANNEL_COUNT:
            raise ValueError(f"PathPilot observations require {CHANNEL_COUNT} channels")
        if self.action_count != len(NavigationAction):
            raise ValueError(f"PathPilot requires {len(NavigationAction)} output actions")


class PathPilotCNN(nn.Module):
    """Small CNN that maps grid observations directly to four action logits."""

    def __init__(self, config: CNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or CNNConfig()
        self.config.validate()

        self.features = nn.Sequential(
            nn.Conv2d(
                self.config.input_channels,
                self.config.first_conv_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.config.first_conv_channels,
                self.config.second_conv_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d(
                (self.config.pooled_height, self.config.pooled_width)
            ),
        )
        flattened_features = (
            self.config.second_conv_channels
            * self.config.pooled_height
            * self.config.pooled_width
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, self.config.hidden_features),
            nn.ReLU(),
            nn.Linear(self.config.hidden_features, self.config.action_count),
        )

    def forward(self, observations: Tensor) -> Tensor:
        """Return unnormalized action scores for a batch of observations."""
        if observations.ndim != 4:
            raise ValueError("CNN input must have shape (batch, channels, height, width)")
        if observations.shape[1] != self.config.input_channels:
            raise ValueError(
                f"CNN input must contain {self.config.input_channels} channels"
            )
        return self.classifier(self.features(observations))

    def predict_action(self, observation: Tensor) -> NavigationAction:
        """Select one action without autograd or unnecessary softmax work."""
        if observation.ndim == 3:
            observation = observation.unsqueeze(0)
        if observation.shape[0] != 1:
            raise ValueError("predict_action expects exactly one observation")

        was_training = self.training
        self.eval()
        with torch.inference_mode():
            action_index = int(self(observation).argmax(dim=1).item())
        self.train(was_training)
        return NavigationAction(action_index)

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable values used by the model."""
        return sum(parameter.numel() for parameter in self.parameters())
