from __future__ import annotations

import argparse
from pathlib import Path

from .config import TrainConfig
from .trainer import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train personal color multitask model.")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--train-csv", default="train_split.csv")
    parser.add_argument("--val-csv", default="val_split.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--export-model-path", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        num_workers=args.num_workers,
        patience=args.patience,
    )
    train(config=config, project_dir=args.project_dir, export_model_path=args.export_model_path)


if __name__ == "__main__":
    main()

