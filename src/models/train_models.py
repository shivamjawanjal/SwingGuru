"""
Phase 5 — Machine Learning

Trains and compares five model families on the SAME chronological
train/val split (train models on train, select the winner on val;
test stays untouched for Phase 6):

  - Logistic Regression  (baseline — linear, one-hot categoricals, scaled)
  - Random Forest         (tree ensemble, one-hot categoricals)
  - XGBoost               (gradient boosting, native categorical support)
  - LightGBM              (gradient boosting, native categorical support)
  - CatBoost              (gradient boosting, native categorical support)

Per the doc: "Don't start with deep learning. Tree models usually win
on tabular financial data." This trains the cheap linear baseline
first specifically so you have a number to confirm the tree models are
actually earning their complexity, not just matching a coin flip with
extra steps.

Model selection uses validation ROC-AUC as the primary metric, but
precision@top-K is computed and reported for every model since that's
what the daily scanner product actually depends on.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config
from src.models.data_loader import load_dataset, chronological_split, prepare_features
from src.models.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_models")


def _train_logistic_regression(data: dict):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE,
        )),
    ])
    model.fit(data["X_train"], data["y_train"])
    return model


def _train_random_forest(data: dict):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", n_jobs=config.MODEL_TRAINING_WORKERS,
        random_state=config.RANDOM_STATE,
    )
    model.fit(data["X_train"], data["y_train"])
    return model


def _train_xgboost(data: dict):
    from xgboost import XGBClassifier
    n_pos = data["y_train"].sum()
    n_neg = len(data["y_train"]) - n_pos
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=config.RANDOM_STATE,
        n_jobs=config.MODEL_TRAINING_WORKERS,
    )
    model.fit(data["X_train"], data["y_train"])
    return model


def _train_lightgbm(data: dict, categorical_columns):
    from lightgbm import LGBMClassifier
    n_pos = data["y_train"].sum()
    n_neg = len(data["y_train"]) - n_pos
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    model = LGBMClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=config.RANDOM_STATE, verbose=-1,
        n_jobs=config.MODEL_TRAINING_WORKERS,
    )
    model.fit(data["X_train"], data["y_train"], categorical_feature=categorical_columns)
    return model


def _train_catboost(data: dict, categorical_columns):
    from catboost import CatBoostClassifier
    n_pos = data["y_train"].sum()
    n_neg = len(data["y_train"]) - n_pos
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    cat_idx = [data["X_train"].columns.get_loc(c) for c in categorical_columns]
    model = CatBoostClassifier(
        iterations=400, depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=config.RANDOM_STATE, verbose=False,
    )
    model.fit(data["X_train"], data["y_train"], cat_features=cat_idx)
    return model


def uses_native_categorical(model_name: str) -> bool:
    return model_name in ("lightgbm", "catboost")


def train_single_model(model_name: str, data: dict, categorical_columns):
    """
    Public single-model trainer — the same code path Phase 5's
    comparison loop uses, exposed so Phase 6's walk-forward validation
    retrains the identical model type/hyperparameters on each fold
    without re-implementing (and risking drifting from) the training
    logic.
    """
    if model_name == "logistic_regression":
        return _train_logistic_regression(data)
    elif model_name == "random_forest":
        return _train_random_forest(data)
    elif model_name == "xgboost":
        return _train_xgboost(data)
    elif model_name == "lightgbm":
        return _train_lightgbm(data, categorical_columns)
    elif model_name == "catboost":
        return _train_catboost(data, categorical_columns)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def train_and_compare(n_workers: Optional[int] = None, selection_metric: Optional[str] = None) -> dict:
    if n_workers is not None:
        config.MODEL_TRAINING_WORKERS = n_workers
    if selection_metric is not None:
        config.MODEL_SELECTION_METRIC = selection_metric

    logger.info("Loading dataset...")
    df, manifest = load_dataset()
    feature_columns = manifest["feature_columns"]
    categorical_columns = manifest["categorical_columns"]
    label_column = manifest["label_column"]

    logger.info("Splitting chronologically (train/val/test, NEVER random)...")
    train_df, val_df, test_df, split_info = chronological_split(df)
    logger.info("Split: %s", split_info)

    if train_df["date"].max() >= val_df["date"].min():
        raise RuntimeError("Chronological split invariant violated — train overlaps val in time.")

    # Two parallel feature preps: one-hot for LR/RF/XGBoost, native categorical for LightGBM/CatBoost.
    data_onehot = prepare_features(
        train_df, val_df, test_df, feature_columns, categorical_columns, label_column,
        one_hot_categoricals=True,
    )
    data_native = prepare_features(
        train_df, val_df, test_df, feature_columns, categorical_columns, label_column,
        one_hot_categoricals=False,
    )

    results = {}
    fitted_models = {}

    model_jobs = [
        ("logistic_regression", lambda: _train_logistic_regression(data_onehot), data_onehot),
        ("random_forest", lambda: _train_random_forest(data_onehot), data_onehot),
        ("xgboost", lambda: _train_xgboost(data_onehot), data_onehot),
        ("lightgbm", lambda: _train_lightgbm(data_native, categorical_columns), data_native),
        ("catboost", lambda: _train_catboost(data_native, categorical_columns), data_native),
    ]

    for name, train_fn, data in model_jobs:
        logger.info("Training %s...", name)
        t0 = time.monotonic()
        try:
            model = train_fn()
        except Exception:
            logger.exception("%s FAILED to train, skipping", name)
            continue
        elapsed = time.monotonic() - t0

        y_val_proba = model.predict_proba(data["X_val"])[:, 1]
        val_metrics = compute_metrics(data["y_val"], y_val_proba)
        val_metrics["train_seconds"] = round(elapsed, 1)

        logger.info("%s val metrics: %s", name, val_metrics)
        results[name] = val_metrics
        fitted_models[name] = (model, data["feature_names"])

    if not results:
        raise RuntimeError("Every model failed to train — check logs above.")

    ranked = sorted(
        results.items(),
        key=lambda kv: (kv[1][config.MODEL_SELECTION_METRIC] if kv[1][config.MODEL_SELECTION_METRIC] is not None else -1),
        reverse=True,
    )
    best_name = ranked[0][0]
    best_model, best_feature_names = fitted_models[best_name]
    logger.info("Best model by %s: %s (%s)", config.MODEL_SELECTION_METRIC, best_name, results[best_name])

    # Save the winner + everything needed to reproduce its predictions.
    from sklearn.calibration import CalibratedClassifierCV
    logger.info("Calibrating model %s probabilities on validation set...", best_name)
    best_data = data_native if uses_native_categorical(best_name) else data_onehot
    
    calibrated_model = CalibratedClassifierCV(estimator=best_model, cv="prefit", method="sigmoid")
    calibrated_model.fit(best_data["X_val"], best_data["y_val"])
    
    joblib.dump(calibrated_model, config.BEST_MODEL_FILE)

    # Save a training features sample to serve as background data for SHAP (linear models)
    bg_sample = data_onehot["X_train"].sample(
        n=min(len(data_onehot["X_train"]), 200), random_state=config.RANDOM_STATE
    )
    joblib.dump(bg_sample, config.MODELS_DIR / "bg_sample.joblib")
    meta = {
        "model_name": best_name,
        "primary_metric": config.MODEL_SELECTION_METRIC,
        "uses_native_categorical": best_name in ("lightgbm", "catboost"),
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "feature_names_after_encoding": best_feature_names,
        "val_metrics": results[best_name],
        "split_info": split_info,
        "trained_at": pd.Timestamp.now().isoformat(),
    }
    with open(config.BEST_MODEL_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    comparison = {
        "primary_metric": config.MODEL_SELECTION_METRIC,
        "best_model": best_name,
        "split_info": split_info,
        "results": results,
    }
    with open(config.MODEL_COMPARISON_FILE, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info("Saved best model to %s, comparison report to %s", config.BEST_MODEL_FILE, config.MODEL_COMPARISON_FILE)
    return comparison


if __name__ == "__main__":
    train_and_compare()
