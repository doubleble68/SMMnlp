"""Binary metrics shared by note, patient, review-flag, and date-presence tasks."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    cm: np.ndarray
    auc_roc: float
    au_prc: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "confusion_matrix": self.cm.tolist(),
            "auc_roc": float(self.auc_roc),
            "au_prc": float(self.au_prc),
        }


def calculate_binary_metrics(y_true: Any, y_pred: Any) -> BinaryMetrics:
    """Calculate binary metrics, using hard labels as curve inputs.

    au_prc is sklearn average precision. These hard-label
    curve summaries do not measure probability calibration or full ranking.
    Sklearn's warnings and errors are intentionally left visible.
    """
    return BinaryMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred),
        recall=recall_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        cm=confusion_matrix(y_true, y_pred),
        auc_roc=roc_auc_score(y_true, y_pred),
        au_prc=average_precision_score(y_true, y_pred),
    )


def print_metrics(result: BinaryMetrics) -> None:
    for label, value in (
        ("Accuracy", result.accuracy),
        ("Precision", result.precision),
        ("Recall", result.recall),
        ("F1-Score", result.f1),
        ("AUC-ROC", result.auc_roc),
        ("AU-PRC", result.au_prc),
    ):
        print(f"{label}: {value:.4f}")


def confusion_labels(reference_positive: Any, predicted_positive: Any) -> np.ndarray:
    """Assign TP, TN, FN, and FP outcome labels."""
    return np.select(
        [
            reference_positive & predicted_positive,
            ~reference_positive & ~predicted_positive,
            reference_positive & ~predicted_positive,
            ~reference_positive & predicted_positive,
        ],
        ["TP", "TN", "FN", "FP"],
        default="Unknown",
    )
