import tempfile
import unittest
from pathlib import Path

import torch

from pathpilot.learning.checkpoints import load_checkpoint, save_checkpoint
from pathpilot.learning.model import PathPilotCNN
from pathpilot.learning.training import TrainingConfig


class CheckpointTests(unittest.TestCase):
    def test_restored_model_produces_identical_logits(self) -> None:
        torch.manual_seed(3)
        model = PathPilotCNN()
        observation = torch.rand((1, 3, 15, 20))
        model.eval()
        with torch.inference_mode():
            expected = model(observation)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(
                path,
                model,
                state_dict=model.state_dict(),
                epoch=4,
                validation_accuracy=0.75,
                training_config=TrainingConfig(),
                dataset_manifest={"config": {"seed": 42}},
            )
            loaded = load_checkpoint(path)
            with torch.inference_mode():
                actual = loaded.model(observation)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(loaded.epoch, 4)
        self.assertEqual(loaded.validation_accuracy, 0.75)

    def test_unknown_checkpoint_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"checkpoint_version": 999}, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)
