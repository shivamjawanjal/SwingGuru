"""
Phase 4 / core — merge one symbol's feature file + label file into
ML-ready rows, with absolute-price columns normalized to %-of-close so
they're comparable once pooled across symbols of very different price
scales.
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config


def _normalize_absolute_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds %-of-close counterparts for columns in
    config.COLUMNS_NEEDING_NORMALIZATION. Raw columns are left intact
    (useful for debugging) — only the normalized versions go into
    DATASET_FEATURE_COLUMNS.
    """
    df = df.copy()
    for col in config.COLUMNS_NEEDING_NORMALIZATION:
        if col not in df.columns:
            continue
        df[f"{col}_pct"] = df[col] / df["close"] * 100
    return df


def build_symbol_dataset(
    symbol: str,
    features_path: Path,
    labels_path: Path,
    cap_bucket: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Returns a merged, normalized, column-selected dataframe for one
    symbol, or None if either input is missing/unusable.
    """
    if not features_path.exists() or not labels_path.exists():
        return None

    features = pd.read_csv(features_path, parse_dates=["date"])
    labels = pd.read_csv(
        labels_path,
        parse_dates=["date", "entry_date", "exit_date"],
    )

    merged = pd.merge(features, labels, on="date", how="inner", validate="one_to_one")

    # Drop censored rows (label is NaN — outcome unknown, not negative).
    merged = merged.dropna(subset=[config.DATASET_LABEL_COLUMN])
    if merged.empty:
        return None

    merged = _normalize_absolute_columns(merged)

    merged.insert(0, "symbol", symbol)
    merged["cap_bucket"] = cap_bucket if cap_bucket is not None else "UNKNOWN"

    feature_cols_present = [c for c in config.DATASET_FEATURE_COLUMNS if c in merged.columns]
    # Drop indicator warm-up rows (e.g. first ~20 rows per symbol where
    # Bollinger/RSI/ATR/support-resistance are still NaN). These slipped
    # through Phase 4's original label-only dropna — a feature-column NaN
    # here would otherwise reach sklearn models that don't tolerate NaN.
    before = len(merged)
    merged = merged.dropna(subset=feature_cols_present)
    if merged.empty:
        return None

    keep_cols = (
        ["symbol", "date", "cap_bucket"]
        + [c for c in config.DATASET_FEATURE_COLUMNS if c in merged.columns]
        + ["open", "high", "low", "close", "volume",
           "entry_date", "entry_price", "exit_date", "exit_price",
           "exit_reason", "holding_days", "realized_return_pct"]
        + [config.DATASET_LABEL_COLUMN]
    )
    missing = set(config.DATASET_FEATURE_COLUMNS) - set(merged.columns)
    if missing:
        raise RuntimeError(
            f"{symbol}: expected feature columns missing after normalization: {missing}. "
            "Check that Phase 2 feature names still match config.DATASET_FEATURE_COLUMNS."
        )

    return merged[keep_cols]
