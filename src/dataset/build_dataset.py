"""
Phase 4 — Build Dataset

Merges every symbol's features + labels into one master table and
writes it to data/datasets/master_dataset.parquet, plus a
dataset_manifest.json describing exactly which columns are features,
which are categorical, which is the label, and which are metadata-only
— so Phase 5 (model training) never has to guess or hardcode column
names.

Per-symbol merging is independent, so runs across a multiprocessing.Pool.
"""

import json
import logging
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config
from src.dataset.merge_symbol import build_symbol_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_dataset")


def _load_cap_buckets() -> dict:
    """symbol -> primary_index (cap bucket), from whichever universe file exists."""
    src = config.VALIDATED_SYMBOLS_FILE if config.VALIDATED_SYMBOLS_FILE.exists() else config.SYMBOLS_MASTER_FILE
    if not src.exists():
        logger.warning("No symbols file found (%s) — cap_bucket will be UNKNOWN for all rows.", src)
        return {}
    df = pd.read_csv(src)
    if "primary_index" not in df.columns:
        return {}
    return dict(zip(df["symbol"].astype(str), df["primary_index"].astype(str)))


def _process_one_symbol(args) -> tuple:
    symbol, cap_bucket = args
    features_path = config.FEATURES_DIR / f"{symbol}.csv"
    labels_path = config.LABELS_DIR / f"{symbol}.csv"
    try:
        df = build_symbol_dataset(symbol, features_path, labels_path, cap_bucket)
    except Exception as exc:
        return symbol, None, f"failed: {exc}"

    if df is None or df.empty:
        return symbol, None, "no usable rows (missing files or all censored)"

    return symbol, df, "ok"


def build_dataset(n_workers: Optional[int] = None) -> dict:
    if not config.FEATURES_DIR.exists() or not any(config.FEATURES_DIR.glob("*.csv")):
        raise RuntimeError(f"No feature files in {config.FEATURES_DIR}. Run Phase 2 first.")
    if not config.LABELS_DIR.exists() or not any(config.LABELS_DIR.glob("*.csv")):
        raise RuntimeError(f"No label files in {config.LABELS_DIR}. Run Phase 3 first.")

    cap_buckets = _load_cap_buckets()
    symbols = sorted(p.stem for p in config.FEATURES_DIR.glob("*.csv"))
    n_workers = n_workers or config.DATASET_BUILD_WORKERS or max(1, cpu_count() - 1)

    logger.info("Merging features+labels for %d symbols with %d workers...", len(symbols), n_workers)

    tasks = [(sym, cap_buckets.get(sym)) for sym in symbols]
    frames = []
    ok, skipped, failed = 0, 0, 0
    issues = []

    with Pool(processes=n_workers) as pool:
        for i, (symbol, df, status) in enumerate(pool.imap_unordered(_process_one_symbol, tasks), 1):
            if status == "ok":
                ok += 1
                frames.append(df)
            elif status.startswith("no usable"):
                skipped += 1
                issues.append((symbol, status))
            else:
                failed += 1
                issues.append((symbol, status))

            if i % 100 == 0 or i == len(tasks):
                logger.info("Progress: %d/%d | ok=%d skipped=%d failed=%d", i, len(tasks), ok, skipped, failed)

    if not frames:
        raise RuntimeError("No symbol produced usable rows — nothing to write.")

    logger.info("Concatenating %d symbol frames...", len(frames))
    master = pd.concat(frames, ignore_index=True)
    master = master.sort_values(["date", "symbol"]).reset_index(drop=True)

    master.to_parquet(config.MASTER_DATASET_FILE, index=False)

    class_counts = master[config.DATASET_LABEL_COLUMN].value_counts().to_dict()
    n_positive = int(class_counts.get(1.0, 0))
    n_total = len(master)
    base_rate = n_positive / n_total * 100 if n_total else 0.0

    manifest = {
        "n_rows": n_total,
        "n_symbols": master["symbol"].nunique(),
        "date_range": [str(master["date"].min().date()), str(master["date"].max().date())],
        "feature_columns": config.DATASET_FEATURE_COLUMNS,
        "categorical_columns": config.DATASET_CATEGORICAL_COLUMNS,
        "label_column": config.DATASET_LABEL_COLUMN,
        "meta_columns": [c for c in config.DATASET_META_COLUMNS if c in master.columns],
        "base_positive_rate_pct": round(base_rate, 2),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "labeling_params": {
            "profit_pct": config.TRIPLE_BARRIER_PROFIT_PCT,
            "stop_pct": config.TRIPLE_BARRIER_STOP_PCT,
            "max_days": config.TRIPLE_BARRIER_MAX_DAYS,
            "entry_lag_days": config.ENTRY_LAG_DAYS,
        },
    }
    with open(config.DATASET_MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    if issues:
        logger.info("Sample issues:")
        for symbol, status in issues[:15]:
            logger.info("  %s: %s", symbol, status)

    logger.info(
        "Dataset build complete: %d rows, %d symbols, base positive rate %.1f%%. "
        "Written to %s (manifest: %s)",
        n_total, manifest["n_symbols"], base_rate,
        config.MASTER_DATASET_FILE, config.DATASET_MANIFEST_FILE,
    )
    return manifest


if __name__ == "__main__":
    build_dataset()
