"""PyTorch adapter for PathPilot's library-independent saved datasets."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from pathpilot.learning.dataset import NavigationDataset


class PathPilotTorchDataset(Dataset[tuple[Tensor, Tensor]]):
    """Expose validated navigation samples through PyTorch's Dataset API.

    The compressed file is loaded once. Individual observations are converted
    to tensors only when requested, avoiding a second full in-memory copy.
    """

    def __init__(self, dataset: NavigationDataset) -> None:
        dataset.validate()
        self._dataset = dataset

    @classmethod
    def from_file(cls, path: str | Path) -> PathPilotTorchDataset:
        """Load a saved PathPilot dataset and wrap it for PyTorch."""
        return cls(NavigationDataset.load(path))

    def __len__(self) -> int:
        return len(self._dataset.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self._dataset.samples[index]
        observation = torch.tensor(sample.observation, dtype=torch.float32)
        action = torch.tensor(int(sample.action), dtype=torch.int64)
        return observation, action

    @property
    def width(self) -> int:
        return self._dataset.width

    @property
    def height(self) -> int:
        return self._dataset.height
