"""
Phase 5 / metrics — classification metrics, with special emphasis on
precision@top-K. This system's actual product is a ranked shortlist
("Top 20 Swing Trades"), so how good the top-K predictions are matters
far more than overall accuracy on a class-imbalanced problem.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config


def precision_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
    """Of the top-K highest-probability predictions, what fraction were actually positive?"""
    if len(y_true) == 0:
        return float("nan")
    k = min(k, len(y_true))
    top_k_idx = np.argsort(-y_proba)[:k]
    return float(y_true[top_k_idx].mean())


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    # AUC/AP need both classes present; guard against a degenerate split.
    if len(np.unique(y_true)) < 2:
        auc, ap = float("nan"), float("nan")
    else:
        auc = roc_auc_score(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)

    return {
        "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
        "average_precision": round(ap, 4) if not np.isnan(ap) else None,
        "precision_at_0.5": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_at_0.5": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_at_0.5": round(f1_score(y_true, y_pred, zero_division=0), 4),
        f"precision_at_top_{config.TOP_K_FOR_PRECISION}": round(
            precision_at_k(y_true, y_proba, config.TOP_K_FOR_PRECISION), 4
        ),
        "precision_at_top_50": round(precision_at_k(y_true, y_proba, 50), 4),
        "precision_at_top_100": round(precision_at_k(y_true, y_proba, 100), 4),
        "base_rate": round(float(y_true.mean()), 4),
        "n_samples": int(len(y_true)),
    }
