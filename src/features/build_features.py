"""
Phase 2 — Build Features

Loads each validated symbol's OHLCV file, runs it through every feature
family (trend, momentum, volatility, volume, price structure,
candlestick), and writes one enriched CSV per symbol to data/features/.

Per-symbol processing is fully independent, so this runs across a
multiprocessing.Pool — same pattern as build_ohlcv.py in Phase 1.
"""

import logging
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config
from src.features.trend import add_ema_features
from src.features.momentum import add_momentum_features
from src.features.volatility import add_volatility_features
from src.features.volume import add_volume_features
from src.features.price_structure import add_price_structure_features
from src.features.candlestick import add_candlestick_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_features")


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chains every feature family in a fixed order. Each function returns
    a fresh copy with new columns appended, so composing them is just
    sequential calls.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df = add_ema_features(df)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_price_structure_features(df)
    df = add_candlestick_features(df)
    return df


def _process_one_symbol(args) -> tuple:
    """Worker function: (symbol_csv_path) -> (symbol, status, n_rows)."""
    path_str = args
    path = Path(path_str)
    symbol = path.stem

    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception as exc:
        return symbol, f"read failed: {exc}", 0

    if len(df) < config.MIN_ROWS_FOR_FEATURES:
        return symbol, f"skipped: only {len(df)} rows < {config.MIN_ROWS_FOR_FEATURES}", 0

    try:
        featured = compute_all_features(df)
    except Exception as exc:
        return symbol, f"feature computation failed: {exc}", 0

    out_path = config.FEATURES_DIR / f"{symbol}.csv"
    featured.to_csv(out_path, index=False)
    return symbol, "ok", len(featured)


def build_features(n_workers: Optional[int] = None) -> dict:
    if not config.OHLCV_DIR.exists():
        raise RuntimeError(f"{config.OHLCV_DIR} does not exist. Run Phase 1 first.")

    # Prefer the validated symbol list if it exists (Phase 1 Step 4 output);
    # fall back to every file in data/ohlcv/ otherwise.
    if config.VALIDATED_SYMBOLS_FILE.exists():
        valid_symbols = set(
            pd.read_csv(config.VALIDATED_SYMBOLS_FILE)["symbol"].astype(str)
        )
        ohlcv_files = [
            p for p in sorted(config.OHLCV_DIR.glob("*.csv"))
            if p.stem in valid_symbols
        ]
        logger.info(
            "Using %d validated symbols (of %d files present in %s).",
            len(ohlcv_files), len(list(config.OHLCV_DIR.glob("*.csv"))), config.OHLCV_DIR,
        )
    else:
        ohlcv_files = sorted(config.OHLCV_DIR.glob("*.csv"))
        logger.warning(
            "%s not found — using all %d files in %s unfiltered. "
            "Run Phase 1 Step 4 (validate) first for cleaner input.",
            config.VALIDATED_SYMBOLS_FILE, len(ohlcv_files), config.OHLCV_DIR,
        )

    if not ohlcv_files:
        raise RuntimeError("No OHLCV files to process.")

    n_workers = n_workers or config.FEATURE_BUILD_WORKERS or max(1, cpu_count() - 1)
    logger.info("Building features for %d symbols with %d workers...", len(ohlcv_files), n_workers)

    tasks = [str(p) for p in ohlcv_files]
    ok, skipped, failed = 0, 0, 0
    failures = []

    with Pool(processes=n_workers) as pool:
        for i, (symbol, status, n_rows) in enumerate(pool.imap_unordered(_process_one_symbol, tasks), 1):
            if status == "ok":
                ok += 1
            elif status.startswith("skipped"):
                skipped += 1
            else:
                failed += 1
                failures.append((symbol, status))

            if i % 100 == 0 or i == len(tasks):
                logger.info("Progress: %d/%d | ok=%d skipped=%d failed=%d", i, len(tasks), ok, skipped, failed)

    if failures:
        logger.info("Sample failures:")
        for symbol, status in failures[:15]:
            logger.info("  %s: %s", symbol, status)

    logger.info(
        "Feature build complete: %d ok, %d skipped (thin history), %d failed. Output: %s",
        ok, skipped, failed, config.FEATURES_DIR,
    )
    return {"ok": ok, "skipped": skipped, "failed": failed, "failures": failures}


if __name__ == "__main__":
    build_features()
