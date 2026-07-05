#!/usr/bin/env python3
"""
SHAP Explainability Report — CLI entry point.

Generates a global feature importance report using SHAP on the
validation set, and optionally shows per-prediction drivers for the
top-K ranked candidates.

Usage:
  python run_explain.py                  # global importance report
  python run_explain.py --top-n 30       # also show top-30 drivers
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_explain")


def main():
    parser = argparse.ArgumentParser(description="NSE Swing Trading Pipeline — SHAP Explainability")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top predictions to show drivers for")
    args = parser.parse_args()

    from configs import config
    from src.models.data_loader import load_dataset, chronological_split, prepare_features
    from src.models.train_models import uses_native_categorical
    from src.explainability.shap_explainer import (
        explain_predictions, generate_global_importance,
        save_global_importance, print_global_importance,
    )

    logger.info("=" * 70)
    logger.info("STEP: SHAP Explainability Report")
    logger.info("=" * 70)
    t0 = time.monotonic()

    # Load model and metadata.
    if not config.BEST_MODEL_FILE.exists():
        raise RuntimeError(f"{config.BEST_MODEL_FILE} not found. Run Phase 5 first.")
    with open(config.BEST_MODEL_META_FILE) as f:
        model_meta = json.load(f)
    model = joblib.load(config.BEST_MODEL_FILE)
    model_name = model_meta["model_name"]

    # Load dataset and reproduce the same chronological split.
    df, manifest = load_dataset()
    feature_columns = manifest["feature_columns"]
    categorical_columns = manifest["categorical_columns"]
    label_column = manifest["label_column"]

    train_df, val_df, _, _ = chronological_split(df)

    one_hot = not uses_native_categorical(model_name)
    data = prepare_features(
        train_df, val_df, val_df,  # test_df unused here
        feature_columns, categorical_columns, label_column,
        one_hot_categoricals=one_hot,
    )

    # Use training data as background for LinearExplainer.
    background = data["X_train"] if model_name == "logistic_regression" else None
    feature_names = data["feature_names"]

    # Compute SHAP on validation set.
    result = explain_predictions(
        model=model,
        model_name=model_name,
        X=data["X_val"],
        feature_names=feature_names,
        background_data=background,
    )

    # Global importance.
    importance = generate_global_importance(result["shap_values"], feature_names)
    save_global_importance(importance)
    print_global_importance(importance)

    # Per-prediction drivers for top-N by probability.
    y_val_proba = model.predict_proba(data["X_val"])[:, 1]
    val_with_drivers = pd.DataFrame({
        "probability": y_val_proba,
        "actual_label": data["y_val"],
        "top_drivers": result["top_drivers"],
    })
    val_with_drivers = val_with_drivers.sort_values("probability", ascending=False).head(args.top_n).reset_index(drop=True)

    print()
    print("=" * 90)
    print(f"  TOP {args.top_n} PREDICTIONS — WHY EACH WAS FLAGGED (model={model_name})")
    print("=" * 90)
    header = f"{'#':>3} {'Prob':>7} {'Actual':>7} {'Top Drivers'}"
    print(header)
    print("-" * 90)
    for i, row in val_with_drivers.iterrows():
        actual = "WIN" if row["actual_label"] == 1 else "LOSS"
        print(f"{i+1:>3} {row['probability']*100:>6.1f}% {actual:>7}   {row['top_drivers']}")
    print("=" * 90)
    print()

    # Save full report.
    report_path = config.SHAP_REPORT_DIR / "shap_val_report.csv"
    val_with_drivers.to_csv(report_path, index=False)
    logger.info("SHAP report saved to %s", report_path)
    logger.info("SHAP analysis complete in %.1fs.", time.monotonic() - t0)


if __name__ == "__main__":
    main()
