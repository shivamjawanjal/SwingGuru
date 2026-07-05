"""
Phase 6b — Walk-Forward Validation

A single train/val/test split (Phase 5) only tells you how the model
did in ONE slice of market history — it could be a lucky or unlucky
regime. Walk-forward retrains on an expanding window and evaluates on
each subsequent slice in turn, giving an honest performance estimate
across multiple regimes:

  Fold 1: Train [------]        Test [--]
  Fold 2: Train [---------]     Test    [--]
  Fold 3: Train [------------]  Test        [--]
  ...

Every row in every test fold was predicted by a model that had NEVER
seen that date or anything after it during training — this is what
"no future leakage" actually means in practice, not just splitting
once.

Retrains the SAME model family/hyperparameters that Phase 5 selected
as the winner (via train_models.train_single_model — the identical
code path, not a reimplementation) on each fold, fresh, so this
doesn't just measure Phase 5's model quality but the whole
architecture's stability across time.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config
from src.models.data_loader import load_dataset, fit_encoder, apply_encoder
from src.models.train_models import train_single_model, uses_native_categorical
from src.models.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("walk_forward")


def _compute_fold_boundaries(
    unique_dates: np.ndarray, n_folds: int, min_train_fraction: float
) -> List[Tuple[np.datetime64, np.datetime64, np.datetime64]]:
    """
    Returns a list of (train_end_date, test_start_date, test_end_date)
    tuples. Training window for fold i is always [start, train_end_date]
    — i.e. expanding, never a fixed-size rolling window — so fold 3's
    training data is a strict superset of fold 1's.
    """
    n = len(unique_dates)
    min_train_end_idx = max(1, int(n * min_train_fraction))
    remaining = n - min_train_end_idx
    if remaining < n_folds:
        raise ValueError(
            f"Not enough trading days ({n}) after the {min_train_fraction:.0%} "
            f"minimum training window to carve out {n_folds} folds. Use fewer "
            f"folds, a smaller min_train_fraction, or more data."
        )
    fold_size = remaining // n_folds

    boundaries = []
    train_end_idx = min_train_end_idx
    for i in range(n_folds):
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + fold_size if i < n_folds - 1 else n - 1
        boundaries.append((
            unique_dates[train_end_idx - 1],
            unique_dates[test_start_idx],
            unique_dates[test_end_idx],
        ))
        train_end_idx = test_end_idx + 1

    return boundaries


def run_walk_forward(
    model_name: str = None,
    n_folds: int = None,
    min_train_fraction: float = None,
) -> dict:
    with open(config.BEST_MODEL_META_FILE) as f:
        best_meta = json.load(f)
    model_name = model_name or best_meta["model_name"]
    n_folds = n_folds or config.WALK_FORWARD_N_FOLDS
    min_train_fraction = min_train_fraction or config.WALK_FORWARD_MIN_TRAIN_FRACTION
    one_hot = not uses_native_categorical(model_name)

    logger.info(
        "Walk-forward validation: model=%s, n_folds=%d, min_train_fraction=%.0f%%",
        model_name, n_folds, min_train_fraction * 100,
    )

    df, manifest = load_dataset()
    feature_columns = manifest["feature_columns"]
    categorical_columns = manifest["categorical_columns"]
    label_column = manifest["label_column"]

    unique_dates = np.sort(df["date"].unique())
    boundaries = _compute_fold_boundaries(unique_dates, n_folds, min_train_fraction)

    fold_reports = []
    all_oof_y_true = []
    all_oof_y_proba = []

    for i, (train_end, test_start, test_end) in enumerate(boundaries, 1):
        train_df = df[df["date"] <= train_end]
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]

        if train_df.empty or test_df.empty:
            logger.warning("Fold %d: empty train or test slice, skipping.", i)
            continue

        encoder = fit_encoder(train_df, feature_columns, categorical_columns, one_hot)
        X_train, y_train = apply_encoder(train_df, feature_columns, categorical_columns, label_column, encoder)
        X_test, y_test = apply_encoder(test_df, feature_columns, categorical_columns, label_column, encoder)

        data = {"X_train": X_train, "y_train": y_train}
        t0 = time.monotonic()
        model = train_single_model(model_name, data, categorical_columns)
        elapsed = time.monotonic() - t0

        y_proba = model.predict_proba(X_test)[:, 1]
        fold_metrics = compute_metrics(y_test, y_proba)
        fold_metrics["train_seconds"] = round(elapsed, 1)
        fold_metrics["train_rows"] = len(train_df)
        fold_metrics["test_rows"] = len(test_df)
        fold_metrics["train_end_date"] = str(pd.Timestamp(train_end).date())
        fold_metrics["test_date_range"] = [
            str(pd.Timestamp(test_start).date()), str(pd.Timestamp(test_end).date())
        ]

        logger.info("Fold %d/%d [%s -> %s]: %s", i, n_folds,
                    fold_metrics["test_date_range"][0], fold_metrics["test_date_range"][1],
                    {k: v for k, v in fold_metrics.items() if k in
                     ("roc_auc", "average_precision", f"precision_at_top_{config.TOP_K_FOR_PRECISION}", "n_samples")})

        fold_reports.append(fold_metrics)
        all_oof_y_true.append(y_test)
        all_oof_y_proba.append(y_proba)

    if not fold_reports:
        raise RuntimeError("No folds produced results — check date range and fold configuration.")

    # Pooled out-of-fold metrics: every prediction here came from a model
    # that had never seen that row's date (or anything after it), so this
    # is the honest aggregate estimate — more robust than any single fold.
    oof_y_true = np.concatenate(all_oof_y_true)
    oof_y_proba = np.concatenate(all_oof_y_proba)
    pooled_metrics = compute_metrics(oof_y_true, oof_y_proba)

    # Stability check: how much does performance vary fold-to-fold?
    fold_aucs = [f["roc_auc"] for f in fold_reports if f["roc_auc"] is not None]
    auc_stability = {
        "mean": round(float(np.mean(fold_aucs)), 4) if fold_aucs else None,
        "std": round(float(np.std(fold_aucs)), 4) if fold_aucs else None,
        "min": round(float(np.min(fold_aucs)), 4) if fold_aucs else None,
        "max": round(float(np.max(fold_aucs)), 4) if fold_aucs else None,
    }

    report = {
        "model_name": model_name,
        "n_folds": len(fold_reports),
        "min_train_fraction": min_train_fraction,
        "pooled_out_of_fold_metrics": pooled_metrics,
        "per_fold_auc_stability": auc_stability,
        "fold_reports": fold_reports,
    }

    with open(config.WALK_FORWARD_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Pooled out-of-fold metrics across all %d folds: %s", len(fold_reports), pooled_metrics)
    logger.info("Fold-to-fold AUC stability: %s", auc_stability)
    logger.info("Report written to %s", config.WALK_FORWARD_REPORT_FILE)
    return report


if __name__ == "__main__":
    run_walk_forward()
