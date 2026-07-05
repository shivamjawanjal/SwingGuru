#!/usr/bin/env python3
"""
Phase 6 Orchestrator — CLI entry point.

Runs leakage checks and evaluates the Phase 5 winning model on the
held-out test set for the first time.

Usage:
  python run_phase6.py
"""

import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_phase6")


def main():
    from src.evaluation.evaluate_test import evaluate_on_test

    logger.info("=" * 70)
    logger.info("STEP: Evaluate on Held-Out Test Set")
    logger.info("=" * 70)
    t0 = time.monotonic()
    report = evaluate_on_test()
    logger.info("Evaluation complete in %.1fs.", time.monotonic() - t0)
    logger.info("Model: %s | Test metrics: %s", report["model_name"], report["test_metrics"])


if __name__ == "__main__":
    main()
