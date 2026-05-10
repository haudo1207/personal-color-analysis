from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def calc_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def build_confusion(y_true: list[int], y_pred: list[int], labels: list[int]) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred, labels=labels).tolist()


def overall_macro_f1(metrics: dict) -> float:
    return float(
        np.mean(
            [
                metrics["fitzpatrick"]["f1_macro"],
                metrics["undertone"]["f1_macro"],
                metrics["season"]["f1_macro"],
            ]
        )
    )

