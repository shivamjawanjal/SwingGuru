#!/usr/bin/env python3
"""
Phase 1 Orchestrator — CLI entry point.

Runs, in order:
  1. Universe Selection   (src/universe/fetch_universe.py)
  2. Bhavcopy Download     (src/data/bhavcopy_downloader.py)
  3. Build OHLCV files     (src/data/build_ohlcv.py)
  4. Validation             (src/data/validate.py)

Usage:
  python run_phase1.py                     # run all steps
  python run_phase1.py --steps universe    # run just one step
  python run_phase1.py --steps universe,download
  python run_phase1.py --start 2020-01-01 --end 2024-01-01
  python run_phase1.py --workers 8         # override OHLCV build parallelism
"""

import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_phase1")

STEP_ORDER = ["universe", "download", "build_ohlcv", "validate"]


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main():
    parser = argparse.ArgumentParser(description="NSE Swing Trading Pipeline — Phase 1")
    parser.add_argument(
        "--steps", type=str, default="all",
        help=f"Comma-separated subset of {STEP_ORDER}, or 'all' (default).",
    )
    parser.add_argument("--start", type=_parse_date, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", type=_parse_date, default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Worker processes for the OHLCV build step (default: cpu_count - 1)",
    )
    args = parser.parse_args()

    steps = STEP_ORDER if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    invalid = set(steps) - set(STEP_ORDER)
    if invalid:
        parser.error(f"Unknown step(s): {invalid}. Valid: {STEP_ORDER}")

    logger.info("Running Phase 1 steps: %s", steps)
    t0 = time.monotonic()

    if "universe" in steps:
        _run_step("Universe Selection", _step_universe)

    if "download" in steps:
        _run_step("Bhavcopy Download", lambda: _step_download(args.start, args.end))

    if "build_ohlcv" in steps:
        _run_step("Build OHLCV", lambda: _step_build_ohlcv(args.workers))

    if "validate" in steps:
        _run_step("Validation", _step_validate)

    logger.info("Phase 1 finished in %.1f minutes.", (time.monotonic() - t0) / 60)


def _run_step(label: str, fn):
    logger.info("=" * 70)
    logger.info("STEP: %s", label)
    logger.info("=" * 70)
    t0 = time.monotonic()
    try:
        result = fn()
        logger.info("%s complete in %.1fs. Result: %s", label, time.monotonic() - t0, result)
    except Exception:
        logger.exception("%s FAILED", label)
        raise


def _step_universe():
    from src.universe.fetch_universe import build_symbol_universe
    return build_symbol_universe()


def _step_download(start, end):
    from src.data.bhavcopy_downloader import download_bhavcopy_range
    return download_bhavcopy_range(start=start, end=end)


def _step_build_ohlcv(workers):
    from src.data.build_ohlcv import build_ohlcv_files
    return build_ohlcv_files(n_workers=workers)


def _step_validate():
    from src.data.validate import validate_ohlcv_files
    return validate_ohlcv_files()


if __name__ == "__main__":
    main()
