import tempfile
import unittest
from pathlib import Path

from pathpilot.environment import GridEnvironment
from pathpilot.learning.actions import NavigationAction
from pathpilot.learning.dataset import NavigationDataset
from pathpilot.learning.encoding import encode_environment
from pathpilot.learning.samples import NavigationSample


class NavigationDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        environment = GridEnvironment(width=3, height=2)
        self.sample = NavigationSample(
            maze_id="maze-0001",
            step_index=0,
            observation=encode_environment(environment),
            action=NavigationAction.RIGHT,
        )

    def test_create_accepts_valid_samples(self) -> None:
        dataset = NavigationDataset.create(
            width=3,
            height=2,
            samples=[self.sample],
        )
        self.assertEqual(dataset.samples, (self.sample,))

    def test_invalid_sample_metadata_is_rejected(self) -> None:
        sample = NavigationSample(
            maze_id="",
            step_index=0,
            observation=self.sample.observation,
            action=NavigationAction.RIGHT,
        )
        with self.assertRaises(ValueError):
            NavigationDataset.create(width=3, height=2, samples=[sample])

    def test_save_and_load_round_trip(self) -> None:
        dataset = NavigationDataset.create(
            width=3,
            height=2,
            samples=[self.sample],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "dataset.json.gz"
            dataset.save(path)
            loaded = NavigationDataset.load(path)

        self.assertEqual(loaded, dataset)

    def test_unknown_schema_version_is_rejected(self) -> None:
        import gzip

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "dataset.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as output:
                output.write('{"schema_version":999,"samples":[]}')
            with self.assertRaises(ValueError):
                NavigationDataset.load(path)
