#!/usr/bin/env python3
"""
Phase 9 Orchestrator — CLI entry point.

Runs the risk-managed backtest: ATR-based stops, trailing stops,
1%-risk position sizing, and per-cap-bucket exposure limits — same
test-set signals as Phase 7, different capital management.

Usage:
  python run_phase9.py
  python run_phase9.py --threshold 0.6 --max-positions 15 --max-per-bucket 3
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_phase9")


def main():
    parser = argparse.ArgumentParser(description="NSE Swing Trading Pipeline — Phase 9 (Risk Management)")
    parser.add_argument("--threshold", type=float, default=None, help="Min predicted probability to take a trade")
    parser.add_argument("--max-positions", type=int, default=None, help="Max concurrent open positions")
    parser.add_argument("--max-per-bucket", type=int, default=None, help="Max concurrent positions from the same cap bucket")
    args = parser.parse_args()

    from src.backtest.risk_managed_engine import run_risk_managed_backtest

    logger.info("=" * 70)
    logger.info("STEP: Risk-Managed Backtest")
    logger.info("=" * 70)
    t0 = time.monotonic()
    report = run_risk_managed_backtest(
        probability_threshold=args.threshold,
        max_concurrent_positions=args.max_positions,
        max_per_cap_bucket=args.max_per_bucket,
    )
    logger.info("Risk-managed backtest complete in %.1fs.", time.monotonic() - t0)
    logger.info("Metrics: %s", report["metrics"])


if __name__ == "__main__":
    main()
