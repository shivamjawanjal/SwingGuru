#!/usr/bin/env python3
"""
Yahoo Finance Data Sync

Downloads the latest daily price data for all validated symbols in a single
batch request, appends new price rows, and caps the history of each symbol
at a rolling 250 days to prevent Git repository bloat.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import yfinance as yf

sys.path.append(str(Path(__file__).resolve().parent))
from configs import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("yfinance_update")

ROLLING_WINDOW_DAYS = 250


def main():
    logger.info("=" * 70)
    logger.info("STEP: Updating Price Database (Yahoo Finance)")
    logger.info("=" * 70)

    # 1. Get validated symbols
    validated_path = config.DATA_DIR / "validated_symbols.csv"
    if not validated_path.exists():
        logger.error("validated_symbols.csv does not exist. Run Phase 1 validation first.")
        return

    validated_df = pd.read_csv(validated_path)
    symbols = validated_df["symbol"].tolist()
    logger.info("Found %d validated symbols to update.", len(symbols))

    # 2. Batch download last 5 days of daily data from yfinance
    tickers = [f"{s}.NS" for s in symbols]
    logger.info("Downloading batch data from Yahoo Finance...")
    try:
        batch_data = yf.download(
            tickers=tickers,
            period="5d",
            group_by="ticker",
            progress=False,
            auto_adjust=True
        )
    except Exception as exc:
        logger.error("Failed to download batch price data from Yahoo Finance: %s", exc)
        return

    # 3. Process each symbol
    updated_count = 0
    skipped_count = 0

    for symbol in symbols:
        ticker = f"{symbol}.NS"
        csv_path = config.OHLCV_DIR / f"{symbol}.csv"

        if not csv_path.exists():
            logger.warning("CSV file for %s does not exist under data/ohlcv/, skipping.", symbol)
            continue

        # Load existing CSV
        existing_df = pd.read_csv(csv_path)
        existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.strftime("%Y-%m-%d")
        last_date_str = existing_df.sort_values("date").iloc[-1]["date"]

        # Extract ticker data from batch
        if ticker not in batch_data.columns.levels[0]:
            logger.warning("Ticker %s not found in batch download, skipping.", ticker)
            continue

        ticker_df = batch_data[ticker].dropna(subset=["Close"]).copy()
        if ticker_df.empty:
            skipped_count += 1
            continue

        # Format yfinance columns to match local schema
        ticker_df = ticker_df.reset_index()
        ticker_df = ticker_df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        ticker_df["date"] = pd.to_datetime(ticker_df["date"]).dt.strftime("%Y-%m-%d")

        # Keep only rows newer than the last date in the CSV
        new_rows = ticker_df[ticker_df["date"] > last_date_str].copy()

        if new_rows.empty:
            skipped_count += 1
            # Re-write existing file to enforce rolling 250 days constraint in case it wasn't capped
            if len(existing_df) > ROLLING_WINDOW_DAYS:
                existing_df = existing_df.sort_values("date").tail(ROLLING_WINDOW_DAYS)
                existing_df.to_csv(csv_path, index=False)
            continue

        # Order columns to match exact CSV schema: close, open, date, high, volume, low
        new_rows = new_rows[["close", "open", "date", "high", "volume", "low"]]

        # Combine, sort, and slice to ROLLING_WINDOW_DAYS
        combined_df = pd.concat([existing_df, new_rows], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["date"])
        combined_df = combined_df.sort_values("date").tail(ROLLING_WINDOW_DAYS)

        # Write back to file
        combined_df.to_csv(csv_path, index=False)
        updated_count += 1

    logger.info("Update complete. Updated: %d, Up-to-date/Skipped: %d", updated_count, skipped_count)


if __name__ == "__main__":
    main()
