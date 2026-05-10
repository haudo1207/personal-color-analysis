from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


REQUIRED_COLS = ["fitzpatrick", "undertone", "personal_color"]


def _normalize_fitz_values(df: pd.DataFrame) -> pd.DataFrame:
    if df["fitzpatrick"].max() == 6:
        df["fitzpatrick"] = df["fitzpatrick"] - 1
    return df


def _resolve_image_path(row: pd.Series) -> str:
    if "image_path" in row and isinstance(row["image_path"], str) and row["image_path"].strip():
        return row["image_path"]
    if "file" in row and isinstance(row["file"], str):
        return row["file"]
    return ""


class PersonalColorDataset(Dataset):
    def __init__(self, csv_file: Path, img_size: int, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.img_size = img_size

        for col in REQUIRED_COLS:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        if "image_path" not in self.data.columns and "file" not in self.data.columns:
            raise ValueError("CSV must contain either `image_path` or `file` column")

        self.data["fitzpatrick"] = self.data["fitzpatrick"].astype(int)
        self.data = _normalize_fitz_values(self.data)
        self.data["undertone"] = self.data["undertone"].astype(int)
        self.data["personal_color"] = self.data["personal_color"].astype(int)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Any, dict[str, int]]:
        row = self.data.iloc[idx]
        img_path = _resolve_image_path(row)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.img_size, self.img_size), (255, 255, 255))

        if self.transform:
            image = self.transform(image)

        labels = {
            "fitzpatrick": int(row["fitzpatrick"]),
            "undertone": int(row["undertone"]),
            "personal_color": int(row["personal_color"]),
        }
        return image, labels


def compute_class_weights(
    labels: list[int],
    num_classes: int,
    device: torch.device,
    clip_min: float,
    clip_max: float,
) -> torch.Tensor:
    counter = Counter(labels)
    total = len(labels)
    weights = []

    for class_id in range(num_classes):
        count = counter.get(class_id, 0)
        weight = clip_max if count == 0 else total / (num_classes * count)
        weights.append(weight)

    weights = np.array(weights, dtype=np.float32)
    weights = np.clip(weights, clip_min, clip_max)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_weighted_sampler(train_csv: Path) -> WeightedRandomSampler:
    df = pd.read_csv(train_csv)
    labels = df["personal_color"].astype(int).tolist()
    counter = Counter(labels)
    sample_weights = torch.DoubleTensor([1.0 / counter[label] for label in labels])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

