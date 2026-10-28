import copy
import math
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from pathpilot.learning.model import PathPilotCNN
from pathpilot.learning.training import (
    TrainingConfig,
    create_data_loader,
    evaluate,
    fit,
    train_one_epoch,
)


def make_dataset(sample_count: int = 8) -> TensorDataset:
    observations = torch.rand((sample_count, 3, 8, 8))
    actions = torch.arange(sample_count, dtype=torch.int64) % 4
    return TensorDataset(observations, actions)


class TrainingTests(unittest.TestCase):
    def test_training_changes_model_weights_and_returns_finite_metrics(self) -> None:
        torch.manual_seed(4)
        model = PathPilotCNN()
        before = copy.deepcopy(model.state_dict())
        loader = create_data_loader(make_dataset(), batch_size=4, shuffle=False, seed=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        metrics = train_one_epoch(
            model,
            loader,
            nn.CrossEntropyLoss(),
            optimizer,
            device=torch.device("cpu"),
        )

        self.assertTrue(math.isfinite(metrics.loss))
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)
        self.assertTrue(
            any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
        )

    def test_evaluation_does_not_change_weights(self) -> None:
        model = PathPilotCNN()
        before = copy.deepcopy(model.state_dict())
        loader = create_data_loader(make_dataset(), batch_size=4, shuffle=False, seed=1)
        evaluate(
            model,
            loader,
            nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
        )
        self.assertTrue(
            all(torch.equal(before[name], value) for name, value in model.state_dict().items())
        )

    def test_fit_records_every_epoch_and_best_state(self) -> None:
        dataset = make_dataset()
        result = fit(
            PathPilotCNN(),
            create_data_loader(dataset, batch_size=4, shuffle=True, seed=7),
            create_data_loader(dataset, batch_size=4, shuffle=False, seed=7),
            TrainingConfig(epochs=2, batch_size=4, seed=7),
        )
        self.assertEqual(len(result.history), 2)
        self.assertIn(result.best_epoch, (1, 2))
        self.assertTrue(result.best_state_dict)

    def test_invalid_training_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrainingConfig(epochs=0).validate()
