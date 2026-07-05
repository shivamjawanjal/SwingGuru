#!/usr/bin/env python3
"""
Phase 8 Orchestrator — CLI entry point.

Refreshes data (latest bhavcopy -> OHLCV -> features) and runs the
daily scanner, printing the Top-N ranked swing trade candidates.

Usage:
  python run_phase8.py
  python run_phase8.py --skip-data-refresh          # use existing feature files
  python run_phase8.py --top-n 30 --lookback-days 14
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_phase8")


def main():
    parser = argparse.ArgumentParser(description="NSE Swing Trading Pipeline — Phase 8 (Daily Scanner)")
    parser.add_argument("--top-n", type=int, default=None, help="How many candidates to rank (default: config.SCANNER_TOP_N)")
    parser.add_argument(
        "--skip-data-refresh", action="store_true",
        help="Scan against existing data/features/ files instead of re-downloading bhavcopy first",
    )
    parser.add_argument("--lookback-days", type=int, default=10, help="Days back to check for new bhavcopy when refreshing")
    parser.add_argument("--no-explain", action="store_true", help="Skip SHAP explainability analysis (faster)")
    args = parser.parse_args()

    from src.scanner.daily_scanner import run_daily_scan

    logger.info("=" * 70)
    logger.info("STEP: Daily Scanner")
    logger.info("=" * 70)
    t0 = time.monotonic()
    run_daily_scan(
        top_n=args.top_n,
        skip_data_refresh=args.skip_data_refresh,
        lookback_days=args.lookback_days,
        explain=not args.no_explain,
    )
    logger.info("Scan complete in %.1fs.", time.monotonic() - t0)


if __name__ == "__main__":
    main()
