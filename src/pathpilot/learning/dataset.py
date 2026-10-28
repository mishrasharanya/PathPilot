"""Validated storage for reproducible supervised-navigation examples."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pathpilot.learning.actions import NavigationAction
from pathpilot.learning.encoding import GridEncoding
from pathpilot.learning.samples import NavigationSample

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class NavigationDataset:
    """A fixed-size collection of labeled maze observations."""

    width: int
    height: int
    samples: tuple[NavigationSample, ...]

    @classmethod
    def create(
        cls,
        *,
        width: int,
        height: int,
        samples: Iterable[NavigationSample],
    ) -> NavigationDataset:
        """Build and validate a dataset from any sample iterable."""
        dataset = cls(width=width, height=height, samples=tuple(samples))
        dataset.validate()
        return dataset

    def validate(self) -> None:
        """Raise an error when dimensions or any contained sample are invalid."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Dataset dimensions must be positive")
        for sample in self.samples:
            sample.validate(width=self.width, height=self.height)

    def save(self, path: str | Path) -> None:
        """Write the dataset as schema-versioned gzip-compressed JSON."""
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "width": self.width,
            "height": self.height,
            "samples": [self._serialize_sample(sample) for sample in self.samples],
        }
        with gzip.open(destination, "wt", encoding="utf-8", newline="") as output:
            json.dump(payload, output, separators=(",", ":"), sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> NavigationDataset:
        """Load and validate a dataset created by :meth:`save`."""
        with gzip.open(Path(path), "rt", encoding="utf-8") as source:
            payload = json.load(source)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported dataset schema version")

        samples = tuple(cls._deserialize_sample(item) for item in payload["samples"])
        return cls.create(
            width=int(payload["width"]),
            height=int(payload["height"]),
            samples=samples,
        )

    @staticmethod
    def _serialize_sample(sample: NavigationSample) -> dict[str, Any]:
        return {
            "maze_id": sample.maze_id,
            "step_index": sample.step_index,
            "observation": sample.observation,
            "action": int(sample.action),
        }

    @staticmethod
    def _deserialize_sample(payload: dict[str, Any]) -> NavigationSample:
        observation: GridEncoding = tuple(
            tuple(tuple(float(value) for value in row) for row in channel)
            for channel in payload["observation"]
        )  # type: ignore[assignment]
        return NavigationSample(
            maze_id=str(payload["maze_id"]),
            step_index=int(payload["step_index"]),
            observation=observation,
            action=NavigationAction(int(payload["action"])),
        )
