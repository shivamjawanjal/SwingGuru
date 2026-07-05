"""
SHAP Explainability Module

Provides per-prediction feature importance via SHAP values so the daily
scanner can show WHY each stock was flagged, not just its probability.

Uses TreeExplainer for tree-based models (XGBoost, LightGBM, CatBoost,
Random Forest) and LinearExplainer for logistic regression. TreeExplainer
is exact and fast for trees; LinearExplainer is the principled choice for
linear models (it attributes coefficients correctly rather than using the
model-agnostic KernelExplainer, which would be slow and noisy for no
reason on a linear model).
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import shap

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("shap_explainer")

# Model families that support TreeExplainer.
_TREE_MODELS = {"random_forest", "xgboost", "lightgbm", "catboost"}


def _get_explainer(model, model_name: str, background_data: Optional[pd.DataFrame] = None):
    # If the model is wrapped in CalibratedClassifierCV, extract the base estimator
    if hasattr(model, "calibrated_classifiers_"):
        model = model.calibrated_classifiers_[0].estimator

    if model_name in _TREE_MODELS:
        return shap.TreeExplainer(model)

    if model_name == "logistic_regression":
        # The model is a sklearn Pipeline: [StandardScaler, LogisticRegression].
        # LinearExplainer needs the raw classifier and pre-scaled background.
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]

        if background_data is None:
            raise ValueError(
                "LinearExplainer for logistic_regression requires background_data "
                "(a sample of training features) to compute expected values."
            )
        n = min(len(background_data), config.SHAP_MAX_BACKGROUND_SAMPLES)
        bg_sample = background_data.sample(n=n, random_state=config.RANDOM_STATE)
        bg_scaled = scaler.transform(bg_sample)
        return shap.LinearExplainer(clf, bg_scaled)

    raise ValueError(f"Unsupported model type for SHAP: {model_name}")


def _preprocess_for_shap(model, model_name: str, X: pd.DataFrame) -> np.ndarray:
    """
    Apply any model-specific preprocessing (e.g. StandardScaler for
    logistic regression) so SHAP values are computed on the correct
    input space.
    """
    # If the model is wrapped in CalibratedClassifierCV, extract the base estimator
    if hasattr(model, "calibrated_classifiers_"):
        model = model.calibrated_classifiers_[0].estimator

    if model_name == "logistic_regression":
        scaler = model.named_steps["scaler"]
        return scaler.transform(X)
    return X.values if isinstance(X, pd.DataFrame) else X


def explain_predictions(
    model,
    model_name: str,
    X: pd.DataFrame,
    feature_names: List[str],
    background_data: Optional[pd.DataFrame] = None,
    top_k: Optional[int] = None,
) -> dict:
    """
    Computes SHAP values for each row in X and returns:
      - shap_values: np.ndarray of shape (n_samples, n_features)
      - top_drivers: list of dicts, one per sample, each containing
        the top-K features driving that prediction (signed).
      - feature_names: list of feature names matching the SHAP columns.

    top_k defaults to config.SHAP_TOP_DRIVERS.
    """
    top_k = top_k or config.SHAP_TOP_DRIVERS

    explainer = _get_explainer(model, model_name, background_data)
    X_processed = _preprocess_for_shap(model, model_name, X)

    logger.info("Computing SHAP values for %d samples (%s)...", len(X), model_name)
    shap_values_raw = explainer.shap_values(X_processed)

    # shap_values_raw may be a list [class_0, class_1] for binary classifiers.
    # We want class_1 (positive = "swing trade worthy").
    if isinstance(shap_values_raw, list):
        shap_vals = shap_values_raw[1]
    elif shap_values_raw.ndim == 3:
        shap_vals = shap_values_raw[:, :, 1]
    else:
        shap_vals = shap_values_raw

    # Build per-sample top-K driver strings.
    top_drivers = []
    for i in range(len(shap_vals)):
        row_shap = shap_vals[i]
        # Get indices sorted by absolute SHAP value, descending.
        sorted_idx = np.argsort(-np.abs(row_shap))[:top_k]
        drivers = []
        for idx in sorted_idx:
            sign = "+" if row_shap[idx] >= 0 else "-"
            name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            drivers.append(f"{sign}{name}")
        top_drivers.append(", ".join(drivers))

    return {
        "shap_values": shap_vals,
        "top_drivers": top_drivers,
        "feature_names": feature_names,
    }


def generate_global_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Computes mean |SHAP value| per feature across all samples — the
    standard global importance measure. Returns a DataFrame sorted by
    importance descending.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": feature_names[:len(mean_abs)],
        "mean_abs_shap": np.round(mean_abs, 6),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance["rank"] = range(1, len(importance) + 1)
    importance["cumulative_pct"] = (
        importance["mean_abs_shap"].cumsum() / importance["mean_abs_shap"].sum() * 100
    ).round(1)

    return importance[["rank", "feature", "mean_abs_shap", "cumulative_pct"]]


def save_global_importance(importance: pd.DataFrame) -> Path:
    """Persist global feature importance to CSV."""
    out_path = config.SHAP_REPORT_DIR / "global_feature_importance.csv"
    importance.to_csv(out_path, index=False)
    logger.info("Global SHAP importance written to %s (%d features)", out_path, len(importance))
    return out_path


def print_global_importance(importance: pd.DataFrame, top_n: int = 15) -> None:
    """Pretty-print the top-N most important features."""
    print()
    print("=" * 72)
    print(f"  GLOBAL FEATURE IMPORTANCE (top {min(top_n, len(importance))} by mean |SHAP|)")
    print("=" * 72)
    header = f"{'Rk':>4} {'Feature':<35} {'Mean|SHAP|':>12} {'Cumul%':>8}"
    print(header)
    print("-" * 72)
    for _, row in importance.head(top_n).iterrows():
        print(f"{row['rank']:>4} {row['feature']:<35} {row['mean_abs_shap']:>12.6f} {row['cumulative_pct']:>7.1f}%")
    print("=" * 72)
    print()
