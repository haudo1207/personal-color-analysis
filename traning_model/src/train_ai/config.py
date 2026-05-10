from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch


@dataclass
class TrainConfig:
    seed: int = 42
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 18
    lr: float = 8e-4
    num_workers: int = 2
    patience: int = 5
    season_loss_weight: float = 1.2
    output_dir: str = "outputs"
    train_csv: str = "train_split.csv"
    val_csv: str = "val_split.csv"
    clip_grad_norm: float = 1.0
    class_weight_clip_max: float = 8.0
    class_weight_clip_min: float = 0.3
    weight_decay: float = 1e-4


FITZ_MAP = {0: "Type 1", 1: "Type 2", 2: "Type 3", 3: "Type 4", 4: "Type 5", 5: "Type 6"}
UNDER_MAP = {0: "Cool", 1: "Neutral", 2: "Warm"}
SEASON_MAP = {0: "Spring", 1: "Summer", 2: "Autumn", 3: "Winter"}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

