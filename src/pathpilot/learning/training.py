"""Explicit, reproducible PyTorch training loops for PathPilot policies."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from pathpilot.learning.model import PathPilotCNN


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters needed to reproduce a supervised training run."""

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    seed: int = 42
    device: str = "cpu"

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.device not in {"cpu", "mps", "cuda"}:
            raise ValueError("device must be cpu, mps, or cuda")


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Loss, action accuracy, and elapsed time from one complete pass."""

    loss: float
    accuracy: float
    sample_count: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """Comparable training and validation measurements for one epoch."""

    epoch: int
    train: EpochMetrics
    validation: EpochMetrics


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Full history and best validation model state from a training run."""

    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_validation_accuracy: float
    best_state_dict: dict[str, Tensor]


def seed_everything(seed: int) -> None:
    """Seed random sources used by the current training pipeline."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_data_loader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic single-process loader."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def train_one_epoch(
    model: PathPilotCNN,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> EpochMetrics:
    """Run forward propagation, backpropagation, and optimizer updates."""
    model.train()
    started_at = perf_counter()
    total_loss = 0.0
    correct = 0
    sample_count = 0

    for observations, actions in data_loader:
        observations = observations.to(device)
        actions = actions.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(observations)
        loss = loss_function(logits, actions)
        loss.backward()
        optimizer.step()

        batch_size = actions.shape[0]
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == actions).sum().item())
        sample_count += batch_size

    return _finalize_metrics(total_loss, correct, sample_count, started_at)


def evaluate(
    model: PathPilotCNN,
    data_loader: DataLoader,
    loss_function: nn.Module,
    *,
    device: torch.device,
) -> EpochMetrics:
    """Measure a dataset without recording gradients or changing weights."""
    model.eval()
    started_at = perf_counter()
    total_loss = 0.0
    correct = 0
    sample_count = 0

    with torch.inference_mode():
        for observations, actions in data_loader:
            observations = observations.to(device)
            actions = actions.to(device)
            logits = model(observations)
            loss = loss_function(logits, actions)

            batch_size = actions.shape[0]
            total_loss += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == actions).sum().item())
            sample_count += batch_size

    return _finalize_metrics(total_loss, correct, sample_count, started_at)


def fit(
    model: PathPilotCNN,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainingConfig,
) -> TrainingResult:
    """Train a model and retain the state with best validation accuracy."""
    config.validate()
    seed_everything(config.seed)
    device = torch.device(config.device)
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    history: list[EpochRecord] = []
    best_epoch = 0
    best_accuracy = -1.0
    best_state: dict[str, Tensor] = {}

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device=device,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            loss_function,
            device=device,
        )
        history.append(
            EpochRecord(
                epoch=epoch,
                train=train_metrics,
                validation=validation_metrics,
            )
        )
        if validation_metrics.accuracy > best_accuracy:
            best_epoch = epoch
            best_accuracy = validation_metrics.accuracy
            best_state = copy.deepcopy(model.state_dict())

    return TrainingResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_accuracy=best_accuracy,
        best_state_dict=best_state,
    )


def _finalize_metrics(
    total_loss: float,
    correct: int,
    sample_count: int,
    started_at: float,
) -> EpochMetrics:
    if sample_count == 0:
        raise ValueError("Cannot calculate metrics for an empty dataset")
    return EpochMetrics(
        loss=total_loss / sample_count,
        accuracy=correct / sample_count,
        sample_count=sample_count,
        duration_seconds=perf_counter() - started_at,
    )
