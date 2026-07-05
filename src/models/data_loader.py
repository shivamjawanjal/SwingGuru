"""
Phase 5 / data_loader — load the master dataset, split it
CHRONOLOGICALLY (never randomly — this is the #1 way retail ML trading
projects fool themselves), and prepare feature matrices for each
model family's requirements (some need one-hot encoding, tree models
based on LightGBM/CatBoost can take the categorical column natively).
"""

import json
import sys
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config


def load_dataset() -> Tuple[pd.DataFrame, dict]:
    if not config.MASTER_DATASET_FILE.exists():
        raise RuntimeError(f"{config.MASTER_DATASET_FILE} not found. Run Phase 4 first.")
    df = pd.read_parquet(config.MASTER_DATASET_FILE)
    with open(config.DATASET_MANIFEST_FILE) as f:
        manifest = json.load(f)
    return df, manifest


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = None,
    val_frac: float = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Splits by DATE cutoffs computed from the sorted unique dates in the
    dataset, not by row count and not randomly. Every row for a given
    date lands in the same split. Returns (train, val, test, cutoffs_info).
    """
    train_frac = train_frac if train_frac is not None else config.TRAIN_FRACTION
    val_frac = val_frac if val_frac is not None else config.VAL_FRACTION

    unique_dates = np.sort(df["date"].unique())
    n_dates = len(unique_dates)

    train_end_idx = int(n_dates * train_frac)
    val_end_idx = int(n_dates * (train_frac + val_frac))

    # Guard against degenerate splits on tiny datasets.
    train_end_idx = max(1, min(train_end_idx, n_dates - 2))
    val_end_idx = max(train_end_idx + 1, min(val_end_idx, n_dates - 1))

    train_cutoff = unique_dates[train_end_idx - 1]
    val_cutoff = unique_dates[val_end_idx - 1]

    train_df = df[df["date"] <= train_cutoff]
    val_df = df[(df["date"] > train_cutoff) & (df["date"] <= val_cutoff)]
    test_df = df[df["date"] > val_cutoff]

    info = {
        "train_date_range": [str(pd.Timestamp(unique_dates[0]).date()), str(pd.Timestamp(train_cutoff).date())],
        "val_date_range": [str(pd.Timestamp(train_cutoff).date()), str(pd.Timestamp(val_cutoff).date())],
        "test_date_range": [str(pd.Timestamp(val_cutoff).date()), str(pd.Timestamp(unique_dates[-1]).date())],
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }
    return train_df, val_df, test_df, info


def fit_encoder(
    fit_df: pd.DataFrame,
    feature_columns: List[str],
    categorical_columns: List[str],
    one_hot_categoricals: bool,
) -> dict:
    """
    Fits categorical encoding on ONE dataframe (always the training
    slice) and returns everything needed to apply the exact same
    encoding to any other dataframe later — the single source of truth
    used by Phase 5 training, walk-forward validation, and standalone
    test-set evaluation, so none of them can quietly drift apart.
    """
    if one_hot_categoricals:
        X_fit = fit_df[feature_columns + categorical_columns]
        onehot_columns = list(pd.get_dummies(X_fit, columns=categorical_columns).columns)
        return {"mode": "onehot", "onehot_columns": onehot_columns}
    else:
        categories = {
            col: list(fit_df[col].astype(str).unique()) for col in categorical_columns
        }
        return {"mode": "native", "categories": categories}


def apply_encoder(
    df: pd.DataFrame,
    feature_columns: List[str],
    categorical_columns: List[str],
    label_column: Optional[str],
    encoder: dict,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Applies a previously-fit encoder to any dataframe. Returns (X, y or None)."""
    X = df[feature_columns + categorical_columns].copy()
    y = df[label_column].astype(int).to_numpy() if label_column and label_column in df.columns else None

    if encoder["mode"] == "onehot":
        X_enc = pd.get_dummies(X, columns=categorical_columns)
        X_enc = X_enc.reindex(columns=encoder["onehot_columns"], fill_value=0)
        return X_enc, y
    else:
        for col in categorical_columns:
            cat_type = pd.api.types.CategoricalDtype(categories=encoder["categories"][col])
            X[col] = X[col].astype(str).astype(cat_type)
        return X, y


def prepare_inference_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    categorical_columns: List[str],
    uses_native_categorical: bool,
    expected_columns: List[str],
) -> pd.DataFrame:
    """
    Encodes a single batch of rows (no train/val/test split — used at
    prediction time by the daily scanner) the same way prepare_features
    encodes training data, then reindexes to the EXACT column set the
    model was fit on (expected_columns, from best_model_meta.json).
    Missing columns fill with 0; extras are dropped. This is what lets
    a category unseen at training time show up at scan time without
    crashing prediction.
    """
    X = df[feature_columns + categorical_columns].copy()
    if uses_native_categorical:
        for col in categorical_columns:
            X[col] = X[col].astype(str).astype("category")
    else:
        X = pd.get_dummies(X, columns=categorical_columns)
    return X.reindex(columns=expected_columns, fill_value=0)


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    categorical_columns: List[str],
    label_column: str,
    one_hot_categoricals: bool = True,
) -> dict:
    """
    Builds aligned X/y matrices for train/val/test. Encoding is fit on
    TRAIN ONLY (via fit_encoder) and applied identically to val/test
    (via apply_encoder) — a category unseen in train can't silently
    create a new column or crash prediction at eval time.

    one_hot_categoricals=True  -> for models needing numeric-only input
                                   (Logistic Regression, Random Forest, XGBoost).
    one_hot_categoricals=False -> categorical column cast to pandas
                                   'category' dtype for native handling
                                   (LightGBM, CatBoost).
    """
    encoder = fit_encoder(train_df, feature_columns, categorical_columns, one_hot_categoricals)

    X_train, y_train = apply_encoder(train_df, feature_columns, categorical_columns, label_column, encoder)
    X_val, y_val = apply_encoder(val_df, feature_columns, categorical_columns, label_column, encoder)
    X_test, y_test = apply_encoder(test_df, feature_columns, categorical_columns, label_column, encoder)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_names": list(X_train.columns),
        "encoder": encoder,
    }
