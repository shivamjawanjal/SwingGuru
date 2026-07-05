"""
Phase 3 — Build Labels

Loads each symbol's feature file (Phase 2 output), runs triple-barrier
labeling, and writes date + label columns to data/labels/<SYMBOL>.csv.

Kept separate from the feature files on purpose — Phase 4 (Dataset
Builder) is what joins features + labels into the final ML-ready
rows, matching the doc's phase breakdown. Keeping label generation
isolated also makes it trivial to re-label with different
profit/stop/window parameters without re-running feature engineering.

Per-symbol labeling is independent, so this runs across a
multiprocessing.Pool — same pattern as Phase 1/2.
"""

import logging
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config
from src.labels.triple_barrier import triple_barrier_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_labels")

LABEL_COLUMNS = [
    "date", "entry_date", "entry_price", "exit_date", "exit_price",
    "exit_reason", "holding_days", "realized_return_pct", "label",
]


def _process_one_symbol(args) -> tuple:
    path_str, profit_pct, stop_pct, max_days, entry_lag_days = args
    path = Path(path_str)
    symbol = path.stem

    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception as exc:
        return symbol, f"read failed: {exc}", 0, 0

    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return symbol, f"missing columns: {required - set(df.columns)}", 0, 0

    try:
        labeled = triple_barrier_labels(
            df[["date", "open", "high", "low", "close"]],
            profit_pct=profit_pct, stop_pct=stop_pct,
            max_days=max_days, entry_lag_days=entry_lag_days,
        )
    except Exception as exc:
        return symbol, f"labeling failed: {exc}", 0, 0

    out = labeled[LABEL_COLUMNS]
    out_path = config.LABELS_DIR / f"{symbol}.csv"
    out.to_csv(out_path, index=False)

    n_labeled = int(out["label"].notna().sum())
    n_positive = int((out["label"] == 1.0).sum())
    return symbol, "ok", n_labeled, n_positive


def build_labels(
    profit_pct: Optional[float] = None,
    stop_pct: Optional[float] = None,
    max_days: Optional[int] = None,
    entry_lag_days: Optional[int] = None,
    n_workers: Optional[int] = None,
) -> dict:
    if not config.FEATURES_DIR.exists() or not any(config.FEATURES_DIR.glob("*.csv")):
        raise RuntimeError(f"No feature files found in {config.FEATURES_DIR}. Run Phase 2 first.")

    profit_pct = profit_pct if profit_pct is not None else config.TRIPLE_BARRIER_PROFIT_PCT
    stop_pct = stop_pct if stop_pct is not None else config.TRIPLE_BARRIER_STOP_PCT
    max_days = max_days if max_days is not None else config.TRIPLE_BARRIER_MAX_DAYS
    entry_lag_days = entry_lag_days if entry_lag_days is not None else config.ENTRY_LAG_DAYS
    n_workers = n_workers or config.LABEL_BUILD_WORKERS or max(1, cpu_count() - 1)

    feature_files = sorted(config.FEATURES_DIR.glob("*.csv"))
    logger.info(
        "Labeling %d symbols with %d workers | profit=+%.1f%% stop=-%.1f%% "
        "window=%dd entry_lag=%dd",
        len(feature_files), n_workers, profit_pct * 100, stop_pct * 100,
        max_days, entry_lag_days,
    )

    tasks = [(str(p), profit_pct, stop_pct, max_days, entry_lag_days) for p in feature_files]
    ok, failed = 0, 0
    total_labeled_rows, total_positive_rows = 0, 0
    failures = []

    with Pool(processes=n_workers) as pool:
        for i, (symbol, status, n_labeled, n_positive) in enumerate(
            pool.imap_unordered(_process_one_symbol, tasks), 1
        ):
            if status == "ok":
                ok += 1
                total_labeled_rows += n_labeled
                total_positive_rows += n_positive
            else:
                failed += 1
                failures.append((symbol, status))

            if i % 100 == 0 or i == len(tasks):
                logger.info("Progress: %d/%d | ok=%d failed=%d", i, len(tasks), ok, failed)

    if failures:
        logger.info("Sample failures:")
        for symbol, status in failures[:15]:
            logger.info("  %s: %s", symbol, status)

    base_rate = (total_positive_rows / total_labeled_rows * 100) if total_labeled_rows else 0.0
    logger.info(
        "Label build complete: %d symbols ok, %d failed. %d total labeled rows, "
        "base positive rate = %.1f%%. Output: %s",
        ok, failed, total_labeled_rows, base_rate, config.LABELS_DIR,
    )
    return {
        "ok": ok,
        "failed": failed,
        "failures": failures,
        "total_labeled_rows": total_labeled_rows,
        "total_positive_rows": total_positive_rows,
        "base_positive_rate_pct": round(base_rate, 2),
    }


if __name__ == "__main__":
    build_labels()
