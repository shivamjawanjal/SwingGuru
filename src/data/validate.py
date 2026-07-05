"""
Phase 1 / Step 4 — Validation

Scans every per-symbol OHLCV file and drops anything that would poison
downstream feature engineering / labeling:

  - empty or unreadable files
  - too few trading days (recent IPOs without enough history)
  - too many zero-volume days (suspended / illiquid stocks)
  - broken rows (non-positive prices, high < low, etc.)

Bad files are moved to data/ohlcv_rejected/ (not deleted — you may want
to inspect why something failed) and a clean validated_symbols.csv is
written alongside the master symbols.csv.

This is fast, single-pass, CPU-light work over a few thousand small
CSVs — no concurrency needed here, threading overhead would dwarf the
actual work per file.
"""

import logging
import sys
import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate")

REJECTED_DIR = config.DATA_DIR / "ohlcv_rejected"
VALIDATED_SYMBOLS_FILE = config.DATA_DIR / "validated_symbols.csv"


def _check_file(path: Path) -> Tuple[bool, str]:
    """Returns (is_valid, reason_if_invalid)."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return False, f"unreadable: {exc}"

    if df.empty:
        return False, "empty file"

    required_cols = {"date", "open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        return False, f"missing columns: {required_cols - set(df.columns)}"

    if len(df) < config.MIN_TRADING_DAYS:
        return False, f"insufficient history: {len(df)} rows < {config.MIN_TRADING_DAYS}"

    # Broken rows: non-positive prices, or high/low inconsistent with OHLC.
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    if df[["open", "high", "low", "close"]].isna().any().any():
        return False, "NaN price values present"

    bad_prices = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad_prices.mean() > 0.01:  # allow a tiny fraction of glitchy rows
        return False, f"{bad_prices.sum()} rows with non-positive prices"

    bad_hilo = df["high"] < df["low"]
    if bad_hilo.mean() > 0.01:
        return False, f"{bad_hilo.sum()} rows with high < low"

    zero_volume_ratio = (df["volume"].fillna(0) <= 0).mean()
    if zero_volume_ratio > config.MAX_ALLOWED_ZERO_VOLUME_RATIO:
        return False, f"zero-volume ratio {zero_volume_ratio:.2%} (likely suspended)"

    return True, ""


def validate_ohlcv_files() -> dict:
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)

    ohlcv_files = sorted(config.OHLCV_DIR.glob("*.csv"))
    if not ohlcv_files:
        raise RuntimeError(
            f"No OHLCV files found in {config.OHLCV_DIR}. "
            "Run src/data/build_ohlcv.py first."
        )

    valid_symbols: List[str] = []
    rejections: List[Tuple[str, str]] = []

    for path in ohlcv_files:
        symbol = path.stem
        is_valid, reason = _check_file(path)
        if is_valid:
            valid_symbols.append(symbol)
        else:
            rejections.append((symbol, reason))
            shutil.move(str(path), str(REJECTED_DIR / path.name))

    # Cross-reference against the universe to preserve metadata columns.
    universe_df = pd.read_csv(config.SYMBOLS_MASTER_FILE)
    validated_df = universe_df[universe_df["symbol"].isin(valid_symbols)].copy()
    validated_df.to_csv(VALIDATED_SYMBOLS_FILE, index=False)

    logger.info(
        "Validation complete: %d valid, %d rejected out of %d total.",
        len(valid_symbols), len(rejections), len(ohlcv_files),
    )
    if rejections:
        logger.info("Sample rejections:")
        for symbol, reason in rejections[:15]:
            logger.info("  %s: %s", symbol, reason)

    return {
        "total_checked": len(ohlcv_files),
        "valid": len(valid_symbols),
        "rejected": len(rejections),
        "rejections": rejections,
        "validated_symbols_file": str(VALIDATED_SYMBOLS_FILE),
    }


if __name__ == "__main__":
    validate_ohlcv_files()
