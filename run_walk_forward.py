#!/usr/bin/env python3
"""
Walk-Forward Validation Orchestrator — CLI entry point (Advanced Phase).

Retrains the Phase 5 winning model family across expanding-window
folds spanning the WHOLE dataset, giving an honest multi-regime
performance estimate instead of trusting a single static split.

Usage:
  python run_walk_forward.py
  python run_walk_forward.py --model lightgbm --folds 5
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_walk_forward")


def main():
    parser = argparse.ArgumentParser(description="NSE Swing Trading Pipeline — Walk-Forward Validation")
    parser.add_argument("--model", type=str, default=None, help="Model family to retrain per fold (default: Phase 5's saved winner)")
    parser.add_argument("--folds", type=int, default=None, help="Number of expanding-window folds (default: config.WALK_FORWARD_N_FOLDS)")
    parser.add_argument("--min-train-fraction", type=float, default=None, help="Minimum fraction of history for fold 1's training window")
    args = parser.parse_args()

    from src.evaluation.walk_forward import run_walk_forward

    logger.info("=" * 70)
    logger.info("STEP: Walk-Forward Validation")
    logger.info("=" * 70)
    t0 = time.monotonic()
    report = run_walk_forward(
        model_name=args.model, n_folds=args.folds, min_train_fraction=args.min_train_fraction,
    )
    logger.info("Walk-forward validation complete in %.1fs.", time.monotonic() - t0)
    logger.info("Pooled out-of-fold metrics: %s", report["pooled_out_of_fold_metrics"])
    logger.info("Fold-to-fold AUC stability: %s", report["per_fold_auc_stability"])


if __name__ == "__main__":
    main()
