#!/usr/bin/env python3
"""
Phase 7 Orchestrator — CLI entry point.

Runs the event-driven backtest over the held-out test set using the
Phase 5 winning model's signals.

Usage:
  python run_phase7.py
  python run_phase7.py --threshold 0.6 --max-positions 15
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_phase7")


def main():
    parser = argparse.ArgumentParser(description="NSE Swing Trading Pipeline — Phase 7 (Backtesting)")
    parser.add_argument("--threshold", type=float, default=None, help="Min predicted probability to take a trade")
    parser.add_argument("--max-positions", type=int, default=None, help="Max concurrent open positions")
    args = parser.parse_args()

    from src.backtest.engine import run_backtest

    logger.info("=" * 70)
    logger.info("STEP: Backtest")
    logger.info("=" * 70)
    t0 = time.monotonic()
    report = run_backtest(probability_threshold=args.threshold, max_concurrent_positions=args.max_positions)
    logger.info("Backtest complete in %.1fs.", time.monotonic() - t0)
    logger.info("Metrics: %s", report["metrics"])


if __name__ == "__main__":
    main()
