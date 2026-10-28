"""Train PathPilot's CNN on an A*-generated navigation dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from pathpilot.learning.checkpoints import save_checkpoint
from pathpilot.learning.model import PathPilotCNN
from pathpilot.learning.torch_dataset import PathPilotTorchDataset
from pathpilot.learning.training import (
    TrainingConfig,
    create_data_loader,
    fit,
    seed_everything,
)


def train_from_directory(
    dataset_directory: Path,
    output_directory: Path,
    config: TrainingConfig,
) -> None:
    """Train from saved splits and write the best model plus JSON history."""
    config.validate()
    manifest_path = dataset_directory / "manifest.json"
    with manifest_path.open(encoding="utf-8") as source:
        manifest = json.load(source)

    train_dataset = PathPilotTorchDataset.from_file(
        dataset_directory / "train.json.gz"
    )
    validation_dataset = PathPilotTorchDataset.from_file(
        dataset_directory / "validation.json.gz"
    )
    seed_everything(config.seed)
    model = PathPilotCNN()
    train_loader = create_data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    validation_loader = create_data_loader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )
    result = fit(model, train_loader, validation_loader, config)

    output_directory.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output_directory / "best_model.pt",
        model,
        state_dict=result.best_state_dict,
        epoch=result.best_epoch,
        validation_accuracy=result.best_validation_accuracy,
        training_config=config,
        dataset_manifest=manifest,
    )
    history = {
        "best_epoch": result.best_epoch,
        "best_validation_accuracy": result.best_validation_accuracy,
        "epochs": [asdict(record) for record in result.history],
    }
    with (output_directory / "history.json").open("w", encoding="utf-8") as output:
        json.dump(history, output, indent=2)
        output.write("\n")
    with (output_directory / "training_config.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(asdict(config), output, indent=2, sort_keys=True)
        output.write("\n")

    for record in result.history:
        print(
            f"epoch {record.epoch:>3}: "
            f"train_loss={record.train.loss:.4f} "
            f"train_acc={record.train.accuracy:.3f} "
            f"val_loss={record.validation.loss:.4f} "
            f"val_acc={record.validation.accuracy:.3f}"
        )
    print(
        f"Best validation accuracy {result.best_validation_accuracy:.3f} "
        f"at epoch {result.best_epoch}"
    )
    print(f"Artifacts written to {output_directory.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/default"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/cnn-v1"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_from_directory(
        args.dataset,
        args.output,
        TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
        ),
    )


if __name__ == "__main__":
    main()
