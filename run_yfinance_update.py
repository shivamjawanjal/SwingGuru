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

    # 2. Configure requests session with custom headers & retries to bypass cloud rate-limiting
    import requests
    from urllib3.util import Retry
    from requests.adapters import HTTPAdapter
    import time

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    })
    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))

    tickers = [f"{s}.NS" for s in symbols]
    logger.info("Downloading batch data from Yahoo Finance...")
    try:
        batch_data = yf.download(
            tickers=tickers,
            period="5d",
            group_by="ticker",
            progress=False,
            auto_adjust=True,
            session=session
        )
    except Exception as exc:
        logger.error("Failed to download batch price data from Yahoo Finance: %s", exc)
        return

    # 3. Process each symbol
    updated_count = 0
    skipped_count = 0
    failed_symbols = []

    # Identify any symbols that failed or were empty in the batch
    for symbol in symbols:
        ticker = f"{symbol}.NS"
        
        has_data = False
        if ticker in batch_data.columns.levels[0]:
            ticker_df = batch_data[ticker].dropna(subset=["Close"])
            if not ticker_df.empty:
                has_data = True
                
        if not has_data:
            failed_symbols.append(symbol)

    if failed_symbols:
        logger.info("Attempting to retry %d missing symbols individually...", len(failed_symbols))
        for idx, symbol in enumerate(failed_symbols):
            ticker = f"{symbol}.NS"
            try:
                time.sleep(1.0) # Rate limit cooling period
                logger.info("[%d/%d] Retrying %s...", idx + 1, len(failed_symbols), ticker)
                single_data = yf.download(
                    tickers=[ticker],
                    period="5d",
                    progress=False,
                    auto_adjust=True,
                    session=session
                )
                if not single_data.empty:
                    # Inject single data back into batch_data
                    single_data.columns = pd.MultiIndex.from_product([[ticker], single_data.columns])
                    if batch_data.empty:
                        batch_data = single_data
                    else:
                        batch_data = pd.concat([batch_data, single_data], axis=1)
            except Exception as exc:
                logger.warning("Retry download failed for %s: %s", ticker, exc)

    # 4. Save and trim each symbol to rolling window
    for symbol in symbols:
        ticker = f"{symbol}.NS"
        csv_path = config.OHLCV_DIR / f"{symbol}.csv"

        if not csv_path.exists():
            continue

        existing_df = pd.read_csv(csv_path)
        existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.strftime("%Y-%m-%d")
        last_date_str = existing_df.sort_values("date").iloc[-1]["date"]

        if ticker not in batch_data.columns.levels[0]:
            skipped_count += 1
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
