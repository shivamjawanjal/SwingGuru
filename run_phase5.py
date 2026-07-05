#!/usr/bin/env python3
"""
Phase 5 Orchestrator — CLI entry point.

Trains and compares Logistic Regression, Random Forest, XGBoost,
LightGBM, and CatBoost on a chronological train/val split, saves the
best model + a comparison report.

Usage:
  python run_phase5.py
  python run_phase5.py --workers 8
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_phase5")


def main():
    parser = argparse.ArgumentParser(description="NSE Swing Trading Pipeline — Phase 5 (Machine Learning)")
    parser.add_argument("--workers", type=int, default=None, help="CPU cores for model training (-1 = all)")
    parser.add_argument(
        "--metric", type=str, default=None,
        help="Model selection metric, e.g. roc_auc, average_precision, precision_at_top_20 (default: config.MODEL_SELECTION_METRIC)",
    )
    args = parser.parse_args()

    from src.models.train_models import train_and_compare

    logger.info("=" * 70)
    logger.info("STEP: Train & Compare Models")
    logger.info("=" * 70)
    t0 = time.monotonic()
    result = train_and_compare(n_workers=args.workers, selection_metric=args.metric)
    logger.info("Training complete in %.1fs.", time.monotonic() - t0)
    logger.info("Best model: %s", result["best_model"])
    logger.info("Full comparison: %s", result["results"])


if __name__ == "__main__":
    main()
