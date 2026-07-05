#!/usr/bin/env python3
"""
Phase 3 Orchestrator — CLI entry point.

Runs triple-barrier label generation across all feature files.

Usage:
  python run_phase3.py
  python run_phase3.py --profit 0.08 --stop 0.04 --days 15
  python run_phase3.py --workers 8
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_phase3")


def main():
    parser = argparse.ArgumentParser(description="NSE Swing Trading Pipeline — Phase 3 (Label Generation)")
    parser.add_argument("--profit", type=float, default=None, help="Profit target, e.g. 0.08 for +8%%")
    parser.add_argument("--stop", type=float, default=None, help="Stop loss, e.g. 0.04 for -4%%")
    parser.add_argument("--days", type=int, default=None, help="Max holding window in trading days")
    parser.add_argument("--entry-lag", type=int, default=None, help="Days after signal to enter (default 1 = next open)")
    parser.add_argument("--workers", type=int, default=None, help="Worker processes (default: cpu_count - 1)")
    args = parser.parse_args()

    from src.labels.build_labels import build_labels

    logger.info("=" * 70)
    logger.info("STEP: Build Labels (Triple-Barrier)")
    logger.info("=" * 70)
    t0 = time.monotonic()
    result = build_labels(
        profit_pct=args.profit, stop_pct=args.stop,
        max_days=args.days, entry_lag_days=args.entry_lag,
        n_workers=args.workers,
    )
    logger.info("Build Labels complete in %.1fs. Result: %s", time.monotonic() - t0, result)


if __name__ == "__main__":
    main()
