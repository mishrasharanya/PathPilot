import json
import tempfile
import unittest
from pathlib import Path

from pathpilot.learning.dataset import NavigationDataset
from pathpilot.learning.actions import NavigationAction
from pathpilot.learning.generator import (
    DatasetGenerationConfig,
    generate_datasets,
    generate_labeled_mazes,
    write_datasets,
)


class DatasetGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DatasetGenerationConfig(
            maze_count=10,
            width=8,
            height=6,
            obstacle_density=0.2,
            seed=17,
        )

    def test_generation_is_reproducible(self) -> None:
        first = generate_labeled_mazes(self.config)
        second = generate_labeled_mazes(self.config)
        self.assertEqual(first, second)

    def test_teacher_creates_one_sample_per_path_movement(self) -> None:
        maze = generate_labeled_mazes(self.config)[0]
        self.assertGreater(len(maze.samples), 0)
        self.assertEqual(
            [sample.step_index for sample in maze.samples],
            list(range(len(maze.samples))),
        )
        self.assertTrue(all(sample.maze_id == maze.maze_id for sample in maze.samples))

    def test_splits_keep_maze_ids_disjoint(self) -> None:
        splits = generate_datasets(self.config)
        identifiers = [
            {sample.maze_id for sample in dataset.samples}
            for dataset in (splits.train, splits.validation, splits.test)
        ]
        self.assertEqual([len(group) for group in identifiers], [8, 1, 1])
        self.assertTrue(identifiers[0].isdisjoint(identifiers[1]))
        self.assertTrue(identifiers[0].isdisjoint(identifiers[2]))
        self.assertTrue(identifiers[1].isdisjoint(identifiers[2]))

    def test_generated_samples_are_valid(self) -> None:
        splits = generate_datasets(self.config)
        for dataset in (splits.train, splits.validation, splits.test):
            dataset.validate()

    def test_randomized_endpoints_produce_all_action_labels(self) -> None:
        config = DatasetGenerationConfig(
            maze_count=40,
            width=8,
            height=6,
            obstacle_density=0.2,
            seed=17,
        )
        actions = {
            sample.action
            for maze in generate_labeled_mazes(config)
            for sample in maze.samples
        }
        self.assertEqual(actions, set(NavigationAction))

    def test_writes_loadable_splits_and_manifest(self) -> None:
        splits = generate_datasets(self.config)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_datasets(splits, self.config, output)
            loaded = NavigationDataset.load(output / "train.json.gz")
            manifest = json.loads((output / "manifest.json").read_text())

        self.assertEqual(loaded, splits.train)
        self.assertEqual(manifest["generator"], "astar")
        self.assertEqual(manifest["config"]["seed"], 17)
        self.assertEqual(manifest["splits"]["train"]["maze_count"], 8)

    def test_invalid_config_is_rejected(self) -> None:
        config = DatasetGenerationConfig(maze_count=2)
        with self.assertRaises(ValueError):
            generate_datasets(config)
