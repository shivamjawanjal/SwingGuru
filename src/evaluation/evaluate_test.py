"""
Phase 6 — Evaluation

Loads the model Phase 5 selected, re-derives the exact same
chronological train/val/test split, runs the explicit leakage checks,
and evaluates on the TEST set — which nothing in Phases 1-5 has ever
touched — for the first genuinely unbiased read of how good this
model is.

Also breaks results down by symbol and by cap bucket, since a model
that looks good in aggregate can still be quietly terrible on small
caps or on one particular symbol dragging the average up.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config
from src.models.data_loader import load_dataset, chronological_split, prepare_features
from src.models.metrics import compute_metrics
from src.evaluation.leakage_checks import run_all_checks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_test")


def _breakdown_by(df: pd.DataFrame, y_proba: np.ndarray, group_col: str, min_rows: int = 30) -> dict:
    """Per-group metrics, skipping groups too small to be meaningful."""
    out = {}
    tmp = df.copy()
    tmp["_proba"] = y_proba
    for group_val, group_df in tmp.groupby(group_col):
        if len(group_df) < min_rows:
            continue
        y_true_g = group_df[config.DATASET_LABEL_COLUMN].astype(int).to_numpy()
        y_proba_g = group_df["_proba"].to_numpy()
        if len(np.unique(y_true_g)) < 2:
            continue  # AUC undefined with one class present
        out[str(group_val)] = compute_metrics(y_true_g, y_proba_g)
    return out


def get_test_predictions(train_frac: Optional[float] = None, val_frac: Optional[float] = None) -> tuple:
    """
    Loads the saved model, re-derives the chronological split, runs
    leakage checks, and returns (test_df_with_proba, model_meta,
    split_info, leakage_report) — test_df has a 'proba' column added.
    Shared by evaluate_on_test() (Phase 6) and the backtest engine
    (Phase 7) so both use identical, leakage-checked predictions.
    """
    if not config.BEST_MODEL_FILE.exists():
        raise RuntimeError(f"{config.BEST_MODEL_FILE} not found. Run Phase 5 first.")

    with open(config.BEST_MODEL_META_FILE) as f:
        model_meta = json.load(f)

    model = joblib.load(config.BEST_MODEL_FILE)

    df, manifest = load_dataset()
    train_df, val_df, test_df, split_info = chronological_split(df, train_frac, val_frac)

    leakage_report = run_all_checks(train_df, val_df, test_df)

    data = prepare_features(
        train_df, val_df, test_df,
        model_meta["feature_columns"], model_meta["categorical_columns"],
        config.DATASET_LABEL_COLUMN,
        one_hot_categoricals=not model_meta["uses_native_categorical"],
    )
    X_test = data["X_test"].reindex(columns=model_meta["feature_names_after_encoding"], fill_value=0)

    test_df = test_df.copy()
    test_df["proba"] = model.predict_proba(X_test)[:, 1]

    return test_df, model_meta, split_info, leakage_report


def evaluate_on_test(train_frac: Optional[float] = None, val_frac: Optional[float] = None) -> dict:
    if not config.BEST_MODEL_FILE.exists():
        raise RuntimeError(f"{config.BEST_MODEL_FILE} not found. Run Phase 5 first.")

    with open(config.BEST_MODEL_META_FILE) as f:
        model_meta = json.load(f)

    logger.info("Evaluating model: %s", model_meta["model_name"])

    test_df, model_meta, split_info, leakage_report = get_test_predictions(train_frac, val_frac)

    if split_info != model_meta["split_info"]:
        logger.warning(
            "Freshly computed split does not match the split recorded at training "
            "time (config.TRAIN_FRACTION/VAL_FRACTION may have changed since Phase 5 "
            "ran). Evaluating on the CURRENT split anyway — re-run Phase 5 if you "
            "want the model and split to match exactly.\n  trained-on: %s\n  current:   %s",
            model_meta["split_info"], split_info,
        )

    logger.info("Running leakage checks...")
    logger.info("Leakage checks passed. Regime shift check: %s", leakage_report["regime_shift_check"])

    y_test = test_df[config.DATASET_LABEL_COLUMN].astype(int).to_numpy()
    y_test_proba = test_df["proba"].to_numpy()
    overall_metrics = compute_metrics(y_test, y_test_proba)

    cm = confusion_matrix(y_test, (y_test_proba >= 0.5).astype(int)).tolist()

    by_symbol = _breakdown_by(test_df, y_test_proba, "symbol")
    by_cap_bucket = _breakdown_by(test_df, y_test_proba, "cap_bucket")

    # Sanity flag: test AUC dramatically higher than val AUC (recorded
    # in model_meta) is itself a leakage smell, not just a nice
    # surprise — real models rarely get MORE confident out-of-sample.
    val_auc = model_meta["val_metrics"].get("roc_auc")
    test_auc = overall_metrics.get("roc_auc")
    suspicious_improvement = (
        val_auc is not None and test_auc is not None and (test_auc - val_auc) > 0.10
    )

    report = {
        "model_name": model_meta["model_name"],
        "split_info": split_info,
        "leakage_checks": leakage_report,
        "test_metrics": overall_metrics,
        "val_metrics_for_comparison": model_meta["val_metrics"],
        "suspicious_test_vs_val_improvement": suspicious_improvement,
        "confusion_matrix_at_0.5": {"labels": ["actual_0", "actual_1"], "matrix": cm},
        "breakdown_by_symbol": by_symbol,
        "breakdown_by_cap_bucket": by_cap_bucket,
    }

    with open(config.TEST_EVALUATION_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Test metrics: %s", overall_metrics)
    if suspicious_improvement:
        logger.warning(
            "Test AUC (%.4f) is notably HIGHER than val AUC (%.4f) — this is "
            "unusual for a real out-of-sample result and worth double-checking "
            "for a leak rather than celebrating.", test_auc, val_auc,
        )
    logger.info("Full report written to %s", config.TEST_EVALUATION_REPORT_FILE)
    return report


if __name__ == "__main__":
    evaluate_on_test()
