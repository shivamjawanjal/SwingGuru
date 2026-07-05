#!/usr/bin/env python3
"""
Phase 4 Orchestrator — CLI entry point.

Merges features + labels across all symbols into the master training
dataset.

Usage:
  python run_phase4.py
  python run_phase4.py --workers 8
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_phase4")


def main():
    parser = argparse.ArgumentParser(description="NSE Swing Trading Pipeline — Phase 4 (Dataset Builder)")
    parser.add_argument("--workers", type=int, default=None, help="Worker processes (default: cpu_count - 1)")
    args = parser.parse_args()

    from src.dataset.build_dataset import build_dataset

    logger.info("=" * 70)
    logger.info("STEP: Build Dataset")
    logger.info("=" * 70)
    t0 = time.monotonic()
    result = build_dataset(n_workers=args.workers)
    logger.info("Build Dataset complete in %.1fs.", time.monotonic() - t0)
    logger.info("Manifest: %s", result)


if __name__ == "__main__":
    main()
