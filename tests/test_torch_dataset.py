import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pathpilot.environment import GridEnvironment
from pathpilot.learning.actions import NavigationAction
from pathpilot.learning.dataset import NavigationDataset
from pathpilot.learning.encoding import encode_environment
from pathpilot.learning.samples import NavigationSample
from pathpilot.learning.torch_dataset import PathPilotTorchDataset


class PathPilotTorchDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        environment = GridEnvironment(width=5, height=4)
        samples = tuple(
            NavigationSample(
                maze_id="maze-1",
                step_index=index,
                observation=encode_environment(environment),
                action=NavigationAction.RIGHT,
            )
            for index in range(2)
        )
        self.dataset = NavigationDataset.create(
            width=5,
            height=4,
            samples=samples,
        )

    def test_item_has_model_ready_shape_and_types(self) -> None:
        observation, action = PathPilotTorchDataset(self.dataset)[0]
        self.assertEqual(observation.shape, (3, 4, 5))
        self.assertEqual(observation.dtype, torch.float32)
        self.assertEqual(action.shape, ())
        self.assertEqual(action.dtype, torch.int64)
        self.assertEqual(action.item(), int(NavigationAction.RIGHT))

    def test_data_loader_builds_batches(self) -> None:
        loader = DataLoader(PathPilotTorchDataset(self.dataset), batch_size=2)
        observations, actions = next(iter(loader))
        self.assertEqual(observations.shape, (2, 3, 4, 5))
        self.assertEqual(actions.shape, (2,))

    def test_loads_saved_dataset_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.json.gz"
            self.dataset.save(path)
            dataset = PathPilotTorchDataset.from_file(path)
        self.assertEqual(len(dataset), 2)
        self.assertEqual((dataset.width, dataset.height), (5, 4))
