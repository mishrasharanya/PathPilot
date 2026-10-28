import tempfile
import unittest
from pathlib import Path

import torch

from pathpilot.environment import Direction, GridEnvironment
from pathpilot.learning.checkpoints import save_checkpoint
from pathpilot.learning.encoding import encode_environment
from pathpilot.learning.inference import CachedObservationEncoder, TorchCNNPolicy
from pathpilot.learning.model import PathPilotCNN
from pathpilot.learning.training import TrainingConfig


class CachedObservationEncoderTests(unittest.TestCase):
    def test_cached_tensor_matches_full_encoding_after_changes(self) -> None:
        environment = GridEnvironment(width=5, height=4)
        encoder = CachedObservationEncoder()
        encoder.encode(environment)
        environment.move(Direction.RIGHT)
        environment.add_obstacle((2, 2))
        tensor, latency_ns = encoder.encode(environment)

        expected = torch.tensor(encode_environment(environment)).unsqueeze(0)
        self.assertTrue(torch.equal(tensor, expected))
        self.assertGreaterEqual(latency_ns, 0)

    def test_removed_obstacle_updates_cached_channel(self) -> None:
        environment = GridEnvironment(width=5, height=4, obstacles={(2, 2)})
        encoder = CachedObservationEncoder()
        encoder.encode(environment)
        environment.remove_obstacle((2, 2))
        tensor, _ = encoder.encode(environment)
        self.assertEqual(tensor[0, 0, 2, 2].item(), 0.0)


class TorchCNNPolicyTests(unittest.TestCase):
    def test_checkpoint_is_loaded_and_prediction_is_timed(self) -> None:
        model = PathPilotCNN()
        manifest = {"config": {"width": 5, "height": 4}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(
                path,
                model,
                state_dict=model.state_dict(),
                epoch=1,
                validation_accuracy=0.5,
                training_config=TrainingConfig(),
                dataset_manifest=manifest,
            )
            policy = TorchCNNPolicy(path)
            prediction = policy.predict(GridEnvironment(width=5, height=4))

        self.assertGreaterEqual(prediction.encoding_latency_ns, 0)
        self.assertGreaterEqual(prediction.inference_latency_ns, 0)

    def test_training_grid_size_is_enforced(self) -> None:
        model = PathPilotCNN()
        manifest = {"config": {"width": 20, "height": 15}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(
                path,
                model,
                state_dict=model.state_dict(),
                epoch=1,
                validation_accuracy=0.5,
                training_config=TrainingConfig(),
                dataset_manifest=manifest,
            )
            policy = TorchCNNPolicy(path)
            with self.assertRaises(ValueError):
                policy.validate_environment(GridEnvironment(width=5, height=4))
