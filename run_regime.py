#!/usr/bin/env python3
"""
Market Regime Detection — CLI entry point.

Classifies the current market regime and optionally prints the full
regime history.

Usage:
  python run_regime.py                 # print current regime
  python run_regime.py --history       # print regime for every date
  python run_regime.py --save          # save regime history CSV
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_regime")


def main():
    parser = argparse.ArgumentParser(description="NSE Swing Trading Pipeline — Market Regime Detection")
    parser.add_argument("--history", action="store_true", help="Print full regime history for all dates")
    parser.add_argument("--save", action="store_true", help="Save regime history to CSV")
    args = parser.parse_args()

    from src.regime.regime_detector import (
        classify_current_regime, compute_regime_history,
        regime_distribution, save_regime_history,
    )

    logger.info("=" * 70)
    logger.info("STEP: Market Regime Detection")
    logger.info("=" * 70)

    current = classify_current_regime()
    _print_current_regime(current)

    if args.history or args.save:
        history = compute_regime_history()
        dist = regime_distribution(history)

        if args.history:
            _print_regime_history(history, dist)

        if args.save:
            save_regime_history(history)


def _print_current_regime(regime_info: dict):
    regime = regime_info["regime"]
    emoji = {"BULL": "[+]", "BEAR": "[-]", "HIGH_VOL": "[!]", "SIDEWAYS": "[~]"}.get(regime, "[?]")
    print()
    print("=" * 60)
    print(f"  CURRENT MARKET REGIME: {emoji} {regime}")
    print("=" * 60)
    print(f"  Date:                {regime_info.get('date', 'N/A')}")
    print(f"  Rolling Return (20d): {regime_info.get('rolling_return_pct', 'N/A'):.2f}%")
    print(f"  Rolling Vol (ann.):   {regime_info.get('rolling_vol_annualized_pct', 'N/A'):.2f}%")
    print("=" * 60)
    print()


def _print_regime_history(history, dist):
    print()
    print("=" * 60)
    print("  REGIME DISTRIBUTION")
    print("=" * 60)
    for regime, pct in sorted(dist["percentages"].items(), key=lambda x: -x[1]):
        count = dist["counts"][regime]
        bar = "#" * int(pct / 2)
        print(f"  {regime:<12} {count:>5} days ({pct:>5.1f}%) {bar}")
    print(f"  {'TOTAL':<12} {dist['total_days']:>5} days")
    print("=" * 60)
    print()

    # Print last 20 days of regime history.
    recent = history.tail(20)
    print("  RECENT REGIME HISTORY (last 20 trading days)")
    print("-" * 60)
    header = f"  {'Date':<12} {'Regime':<12} {'Return%':>9} {'Vol%':>9}"
    print(header)
    print("-" * 60)
    for _, row in recent.iterrows():
        d = str(row["date"])[:10]
        emoji = {"BULL": "[+]", "BEAR": "[-]", "HIGH_VOL": "[!]", "SIDEWAYS": "[~]"}.get(row["regime"], "?")
        print(f"  {d:<12} {emoji} {row['regime']:<9} {row['rolling_return_pct']:>8.2f}% {row['rolling_vol_annualized_pct']:>8.2f}%")
    print("-" * 60)
    print()


if __name__ == "__main__":
    main()
