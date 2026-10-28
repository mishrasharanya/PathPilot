"""Versioned saving and loading of trained PathPilot CNN checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from pathpilot.learning.actions import NavigationAction
from pathpilot.learning.model import CNNConfig, PathPilotCNN
from pathpilot.learning.training import TrainingConfig

CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Restored model plus the metadata needed to interpret it."""

    model: PathPilotCNN
    epoch: int
    validation_accuracy: float
    training_config: dict[str, Any]
    dataset_manifest: dict[str, Any]


def save_checkpoint(
    path: str | Path,
    model: PathPilotCNN,
    *,
    state_dict: Mapping[str, torch.Tensor],
    epoch: int,
    validation_accuracy: float,
    training_config: TrainingConfig,
    dataset_manifest: Mapping[str, Any],
) -> None:
    """Save weights and reproducibility metadata in a versioned checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_config": asdict(model.config),
        "model_state_dict": dict(state_dict),
        "epoch": epoch,
        "validation_accuracy": validation_accuracy,
        "training_config": asdict(training_config),
        "dataset_manifest": dict(dataset_manifest),
        "action_labels": {action.name: int(action) for action in NavigationAction},
    }
    torch.save(payload, destination)


def load_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> LoadedCheckpoint:
    """Restore a checkpoint without allowing arbitrary pickled objects."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    expected_labels = {action.name: int(action) for action in NavigationAction}
    if payload.get("action_labels") != expected_labels:
        raise ValueError("Checkpoint action labels do not match PathPilot")

    model = PathPilotCNN(CNNConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.to(torch.device(device))
    model.eval()
    return LoadedCheckpoint(
        model=model,
        epoch=int(payload["epoch"]),
        validation_accuracy=float(payload["validation_accuracy"]),
        training_config=dict(payload["training_config"]),
        dataset_manifest=dict(payload["dataset_manifest"]),
    )
