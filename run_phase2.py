#!/usr/bin/env python3
"""
Phase 2 Orchestrator — CLI entry point.

Runs feature engineering across all validated symbols.

Usage:
  python run_phase2.py
  python run_phase2.py --workers 8
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_phase2")


def main():
    parser = argparse.ArgumentParser(description="NSE Swing Trading Pipeline — Phase 2 (Feature Engineering)")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Worker processes for feature building (default: cpu_count - 1)",
    )
    args = parser.parse_args()

    from src.features.build_features import build_features

    logger.info("=" * 70)
    logger.info("STEP: Build Features")
    logger.info("=" * 70)
    t0 = time.monotonic()
    result = build_features(n_workers=args.workers)
    logger.info("Build Features complete in %.1fs. Result: %s", time.monotonic() - t0, result)


if __name__ == "__main__":
    main()
