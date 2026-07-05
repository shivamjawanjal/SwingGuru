#!/usr/bin/env python3
"""
Paper Trading System - CLI Interface

Saves state, checks exits daily against new data, executes buys at next open,
and prints detailed portfolio reports.

Usage:
  python run_paper_trading.py --report
  python run_paper_trading.py --update
  python run_paper_trading.py --reset
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_paper_trading")


def main():
    parser = argparse.ArgumentParser(description="NSE Swing Trading Pipeline — Paper Trading System")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--report", action="store_true", help="Print current portfolio report and trade history")
    group.add_argument("--update", action="store_true", help="Download new prices, update active exits, execute pending buys, scan new signals")
    group.add_argument("--reset", action="store_true", help="Reset cash and wipe portfolio state")
    
    args = parser.parse_args()

    from src.portfolio.manager import (
        get_portfolio_summary, update_portfolio,
        add_pending_entries, log_daily_equity, reset_portfolio
    )
    from src.scanner.daily_scanner import run_daily_scan, _get_latest_feature_row_per_symbol

    if args.reset:
        confirm = input("Are you sure you want to reset the portfolio state? This wipes all trade history! (y/N): ")
        if confirm.lower() == 'y':
            reset_portfolio()
        else:
            logger.info("Reset cancelled.")
        return

    if args.update:
        logger.info("=" * 70)
        logger.info("STEP: Updating Paper Trading Portfolio")
        logger.info("=" * 70)
        
        # 1. Determine current trading date from the database
        try:
            latest_df = _get_latest_feature_row_per_symbol()
            current_date = str(latest_df["date"].max().date())
        except Exception as exc:
            logger.error("Failed to load feature files to determine latest date: %s", exc)
            logger.error("Please run Phase 2 feature builder first.")
            return

        logger.info("Current execution date is: %s", current_date)
        
        # 2. Update existing active exits and fill pending entries
        update_portfolio(current_date)
        
        # 3. Generate today's scan report for new trade setups
        logger.info("Scanning for new swing trade signals...")
        try:
            scanner_report = run_daily_scan(skip_data_refresh=True, explain=False)
        except Exception as exc:
            logger.error("Daily scan execution failed: %s", exc)
            return

        # 4. Queue candidates from today's scan for T+1 open execution
        add_pending_entries(scanner_report, current_date)
        
        # 5. Log portfolio equity curve progress
        log_daily_equity(current_date)
        
        # 6. Send daily portfolio update alert to Telegram
        try:
            from src.portfolio.manager import get_portfolio_summary
            from src.utils.telegram_alerts import send_telegram_message
            summary = get_portfolio_summary()
            
            p_msg = f"<b>💼 PORTFOLIO UPDATE SUMMARY ({current_date})</b>\n"
            p_msg += f"Total Equity: <b>INR {summary['total_equity']:,.2f}</b>\n"
            p_msg += f"Cash: <code>INR {summary['cash']:,.2f}</code> | Open Value: <code>INR {summary['positions_value']:,.2f}</code>\n"
            p_msg += f"Open Positions: <b>{len(summary['open_positions'])}</b> | Pending Queued: <b>{len(summary['pending_entries'])}</b>\n\n"
            
            if summary["open_positions"]:
                p_msg += "<b>Active Open Positions:</b>\n"
                for pos in summary["open_positions"]:
                    p_msg += f"\u2022 <b>{pos['symbol']}</b>: Px: <code>{pos['current_price']:.2f}</code> | PnL: <b>{pos['pnl_pct']:.2f}%</b> (INR {pos['pnl']:.2f})\n"
                p_msg += "\n"
            
            p_msg += f"Completed Trades: <b>{summary['n_closed_trades']}</b> | Win Rate: <b>{summary['metrics']['win_rate_pct']}%</b>"
            send_telegram_message(p_msg)
        except Exception as e:
            logger.warning("Failed to send Telegram portfolio alert: %s", e)

        logger.info("Portfolio update complete for %s.", current_date)
        return

    if args.report:
        summary = get_portfolio_summary()
        _print_report(summary)


def _print_report(summary: dict):
    print()
    print("=" * 90)
    print("  PAPER TRADING PORTFOLIO SUMMARY")
    print("=" * 90)
    print(f"  Total Portfolio Value (Equity): INR {summary['total_equity']:,.2f}")
    print(f"  Cash Balance:                  INR {summary['cash']:,.2f}")
    print(f"  Open Positions Value:          INR {summary['positions_value']:,.2f}")
    print(f"  Active Open Positions:         {len(summary['open_positions'])}")
    print(f"  Pending Entries queued:        {len(summary['pending_entries'])}")
    print(f"  Completed Trades:              {summary['n_closed_trades']}")
    print("=" * 90)

    # Active Open Positions
    if summary["open_positions"]:
        print()
        print("  ACTIVE OPEN POSITIONS")
        print("-" * 90)
        header = f"  {'Symbol':<12} {'Entry Date':<12} {'Entry Px':>10} {'Curr Px':>10} {'Shares':>8} {'PnL (INR)':>10} {'PnL%':>7}"
        print(header)
        print("-" * 90)
        for pos in summary["open_positions"]:
            print(
                f"  {pos['symbol']:<12} {pos['entry_date']:<12} "
                f"{pos['entry_price']:>10.2f} {pos['current_price']:>10.2f} "
                f"{pos['shares']:>8.2f} {pos['pnl']:>10.2f} {pos['pnl_pct']:>6.2f}%"
            )
        print("-" * 90)

    # Pending queued entries
    if summary["pending_entries"]:
        print()
        print("  PENDING ENTRIES (Queued for next trading session execution at open)")
        print("-" * 90)
        header = f"  {'Symbol':<12} {'Signal Date':<12} {'Prob':>8} {'Cap Bucket':<18}"
        print(header)
        print("-" * 90)
        for entry in summary["pending_entries"]:
            print(f"  {entry['symbol']:<12} {entry['signal_date']:<12} {entry['probability']*100:>6.1f}% {entry['cap_bucket']:<18}")
        print("-" * 90)

    # Completed trade metrics
    metrics = summary["metrics"]
    print()
    print("=" * 90)
    print("  PERFORMANCE METRICS")
    print("=" * 90)
    print(f"  Number of Closed Trades:  {metrics['n_trades']}")
    print(f"  Win Rate:                 {metrics['win_rate_pct']}%")
    print(f"  Total Realized Net PnL:   INR {metrics['total_net_pnl']:,.2f}")
    if metrics['n_trades'] > 0:
        print(f"  Profit Factor:            {metrics['profit_factor']}")
        print(f"  Average Trade Return:     {metrics['expectancy_pct']}%")
        print(f"  Average Holding Period:   {metrics['avg_holding_days']} days")
    print("=" * 90)
    print()


if __name__ == "__main__":
    main()
