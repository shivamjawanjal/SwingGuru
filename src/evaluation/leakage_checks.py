"""
Phase 6 / leakage_checks — explicit, testable invariants for the
train/val/test split. The doc's #1 evaluation priority is "no future
leakage" — this module turns that from a hopeful comment into
assertions that actually run and fail loudly if violated.
"""

import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config


def assert_no_temporal_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Hard invariant: every train date must be strictly before every val
    date, and every val date strictly before every test date. Raises
    AssertionError if violated — this should never pass silently.
    """
    train_max = train_df["date"].max()
    val_min, val_max = val_df["date"].min(), val_df["date"].max()
    test_min = test_df["date"].min()

    assert train_max < val_min, (
        f"LEAKAGE: train max date ({train_max}) is not before val min date ({val_min})"
    )
    assert val_max < test_min, (
        f"LEAKAGE: val max date ({val_max}) is not before test min date ({test_min})"
    )

    return {
        "train_max_date": str(train_max.date()),
        "val_range": [str(val_min.date()), str(val_max.date())],
        "test_min_date": str(test_min.date()),
        "passed": True,
    }


def assert_no_duplicate_rows_across_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Hard invariant: no (symbol, date) row should appear in more than
    one split. With a pure date-cutoff split this should be
    structurally impossible, but we verify it directly rather than
    assuming the split code has no bugs.
    """
    train_keys = set(zip(train_df["symbol"], train_df["date"]))
    val_keys = set(zip(val_df["symbol"], val_df["date"]))
    test_keys = set(zip(test_df["symbol"], test_df["date"]))

    train_val_overlap = train_keys & val_keys
    val_test_overlap = val_keys & test_keys
    train_test_overlap = train_keys & test_keys

    assert not train_val_overlap, f"LEAKAGE: {len(train_val_overlap)} rows duplicated between train and val"
    assert not val_test_overlap, f"LEAKAGE: {len(val_test_overlap)} rows duplicated between val and test"
    assert not train_test_overlap, f"LEAKAGE: {len(train_test_overlap)} rows duplicated between train and test"

    return {"duplicates_found": 0, "passed": True}


def check_regime_shift(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Not a hard failure — a warning. Base positive rate drifting a lot
    between periods doesn't mean there's a bug, but it does mean
    "the market regime changed" is a live possibility worth knowing
    about before trusting the test numbers at face value.
    """
    rates = {
        "train": float(train_df[config.DATASET_LABEL_COLUMN].mean()) * 100,
        "val": float(val_df[config.DATASET_LABEL_COLUMN].mean()) * 100,
        "test": float(test_df[config.DATASET_LABEL_COLUMN].mean()) * 100,
    }
    max_diff = max(rates.values()) - min(rates.values())
    return {
        "base_positive_rate_pct": {k: round(v, 2) for k, v in rates.items()},
        "max_difference_pct_points": round(max_diff, 2),
        "regime_shift_flag": max_diff > config.REGIME_SHIFT_WARNING_THRESHOLD_PCT,
    }


def run_all_checks(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    return {
        "temporal_overlap_check": assert_no_temporal_overlap(train_df, val_df, test_df),
        "duplicate_rows_check": assert_no_duplicate_rows_across_splits(train_df, val_df, test_df),
        "regime_shift_check": check_regime_shift(train_df, val_df, test_df),
    }
