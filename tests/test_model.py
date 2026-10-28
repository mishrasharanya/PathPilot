import unittest

import torch

from pathpilot.learning.actions import NavigationAction
from pathpilot.learning.model import CNNConfig, PathPilotCNN


class PathPilotCNNTests(unittest.TestCase):
    def test_forward_returns_four_logits_per_observation(self) -> None:
        model = PathPilotCNN()
        observations = torch.zeros((8, 3, 15, 20), dtype=torch.float32)
        logits = model(observations)
        self.assertEqual(logits.shape, (8, 4))

    def test_backward_pass_populates_gradients(self) -> None:
        model = PathPilotCNN()
        observations = torch.rand((2, 3, 15, 20))
        loss = model(observations).sum()
        loss.backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.parameters())
        )

    def test_predict_action_uses_single_observation(self) -> None:
        model = PathPilotCNN()
        action = model.predict_action(torch.zeros((3, 15, 20)))
        self.assertIsInstance(action, NavigationAction)

    def test_predict_action_restores_training_mode(self) -> None:
        model = PathPilotCNN()
        model.train()
        model.predict_action(torch.zeros((3, 15, 20)))
        self.assertTrue(model.training)

    def test_model_remains_compact(self) -> None:
        model = PathPilotCNN()
        self.assertLess(model.parameter_count, 50_000)
        self.assertGreater(model.parameter_count, 0)

    def test_invalid_input_shape_is_rejected(self) -> None:
        model = PathPilotCNN()
        with self.assertRaises(ValueError):
            model(torch.zeros((3, 15, 20)))
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 2, 15, 20)))

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PathPilotCNN(CNNConfig(first_conv_channels=0))
