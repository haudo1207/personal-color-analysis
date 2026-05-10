from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from .config import FITZ_MAP, SEASON_MAP, UNDER_MAP, TrainConfig, build_run_dir, get_device
from .dataset import (
    PersonalColorDataset,
    compute_class_weights,
    create_weighted_sampler,
)
from .metrics import build_confusion, calc_metrics, overall_macro_f1
from .model import PersonalColorModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(img_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, val_tf


def build_losses_from_train_csv(train_csv: Path, config: TrainConfig, device: torch.device):
    df = pd.read_csv(train_csv)
    if df["fitzpatrick"].max() == 6:
        df["fitzpatrick"] = df["fitzpatrick"] - 1

    fitz_w = compute_class_weights(
        df["fitzpatrick"].astype(int).tolist(),
        6,
        device,
        config.class_weight_clip_min,
        config.class_weight_clip_max,
    )
    under_w = compute_class_weights(
        df["undertone"].astype(int).tolist(),
        3,
        device,
        config.class_weight_clip_min,
        config.class_weight_clip_max,
    )
    season_w = compute_class_weights(
        df["personal_color"].astype(int).tolist(),
        4,
        device,
        config.class_weight_clip_min,
        config.class_weight_clip_max,
    )
    return (
        nn.CrossEntropyLoss(weight=fitz_w),
        nn.CrossEntropyLoss(weight=under_w),
        nn.CrossEntropyLoss(weight=season_w),
    )


def evaluate(model, loader, criterion_fitz, criterion_under, criterion_season, device):
    model.eval()
    total_loss = 0.0
    true_fitz, pred_fitz = [], []
    true_under, pred_under = [], []
    true_season, pred_season = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            y_fitz = labels["fitzpatrick"].to(device)
            y_under = labels["undertone"].to(device)
            y_season = labels["personal_color"].to(device)

            out_fitz, out_under, out_season = model(images)

            loss_fitz = criterion_fitz(out_fitz, y_fitz)
            loss_under = criterion_under(out_under, y_under)
            loss_season = criterion_season(out_season, y_season)
            total_loss += (loss_fitz + loss_under + loss_season).item()

            pred_fitz.extend(torch.argmax(out_fitz, dim=1).cpu().tolist())
            pred_under.extend(torch.argmax(out_under, dim=1).cpu().tolist())
            pred_season.extend(torch.argmax(out_season, dim=1).cpu().tolist())

            true_fitz.extend(y_fitz.cpu().tolist())
            true_under.extend(y_under.cpu().tolist())
            true_season.extend(y_season.cpu().tolist())

    metrics = {
        "val_loss": float(total_loss / max(len(loader), 1)),
        "fitzpatrick": calc_metrics(true_fitz, pred_fitz),
        "undertone": calc_metrics(true_under, pred_under),
        "season": calc_metrics(true_season, pred_season),
        "confusion_matrix": {
            "fitzpatrick": build_confusion(true_fitz, pred_fitz, list(FITZ_MAP.keys())),
            "undertone": build_confusion(true_under, pred_under, list(UNDER_MAP.keys())),
            "season": build_confusion(true_season, pred_season, list(SEASON_MAP.keys())),
        },
    }
    metrics["overall_macro_f1"] = overall_macro_f1(metrics)
    return metrics


def plot_confusion_matrices(metrics, save_path: Path):
    cm_fitz = np.array(metrics["confusion_matrix"]["fitzpatrick"])
    cm_under = np.array(metrics["confusion_matrix"]["undertone"])
    cm_season = np.array(metrics["confusion_matrix"]["season"])

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    plots = [
        (cm_fitz, "Fitzpatrick", list(FITZ_MAP.values())),
        (cm_under, "Undertone", list(UNDER_MAP.values())),
        (cm_season, "Season", list(SEASON_MAP.values())),
    ]

    for ax, (cm, title, labels) in zip(axes, plots):
        ax.imshow(cm, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, str(cm[row, col]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def train(config: TrainConfig, project_dir: Path, export_model_path: Path | None = None):
    set_seed(config.seed)
    device = get_device()
    run_dir = build_run_dir(project_dir / config.output_dir)
    print(f"[*] Device: {device}")
    print(f"[*] Run dir: {run_dir}")

    train_csv = project_dir / config.train_csv
    val_csv = project_dir / config.val_csv

    train_tf, val_tf = get_transforms(config.img_size)
    train_dataset = PersonalColorDataset(train_csv, img_size=config.img_size, transform=train_tf)
    val_dataset = PersonalColorDataset(val_csv, img_size=config.img_size, transform=val_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=create_weighted_sampler(train_csv),
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = PersonalColorModel().to(device)
    criterion_fitz, criterion_under, criterion_season = build_losses_from_train_csv(train_csv, config, device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_f1 = -1.0
    wait = 0

    best_model_path = run_dir / "best_personal_color_model_full.pth"
    best_metrics_path = run_dir / "evaluation_metrics.json"
    cm_path = run_dir / "confusion_matrices.png"
    history_path = run_dir / "history.json"
    history = []

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            y_fitz = labels["fitzpatrick"].to(device)
            y_under = labels["undertone"].to(device)
            y_season = labels["personal_color"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out_fitz, out_under, out_season = model(images)
                loss_fitz = criterion_fitz(out_fitz, y_fitz)
                loss_under = criterion_under(out_under, y_under)
                loss_season = criterion_season(out_season, y_season)
                total_loss = loss_fitz + loss_under + config.season_loss_weight * loss_season

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += total_loss.item()

        train_loss = running_loss / max(len(train_loader), 1)
        metrics = evaluate(model, val_loader, criterion_fitz, criterion_under, criterion_season, device)
        metrics["train_loss"] = float(train_loss)
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        scheduler.step(metrics["overall_macro_f1"])

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['val_loss']:.4f} | "
            f"Overall Macro F1: {metrics['overall_macro_f1']:.4f}"
        )

        if metrics["overall_macro_f1"] > best_f1:
            best_f1 = metrics["overall_macro_f1"]
            wait = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": "efficientnet_b0",
                    "img_size": config.img_size,
                    "fitz_map": FITZ_MAP,
                    "under_map": UNDER_MAP,
                    "season_map": SEASON_MAP,
                    "best_overall_macro_f1": best_f1,
                },
                best_model_path,
            )
            best_metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
            plot_confusion_matrices(metrics, cm_path)
            print(f"[+] Saved best checkpoint at epoch {epoch + 1}")
        else:
            wait += 1
            if wait >= config.patience:
                print("[*] Early stopping.")
                break

        history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    if export_model_path:
        export_model_path.parent.mkdir(parents=True, exist_ok=True)
        export_model_path.write_bytes(best_model_path.read_bytes())
        print(f"[+] Exported best model to: {export_model_path}")

    print("\n========== TRAINING COMPLETE ==========")
    print(f"best_model: {best_model_path}")
    print(f"metrics: {best_metrics_path}")
    print(f"history: {history_path}")
    print(f"confusion: {cm_path}")

