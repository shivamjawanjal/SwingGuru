#!/usr/bin/env python3
"""
Intraday Position Tracker

Checks open paper trading positions every 5 minutes during market hours
using Yahoo Finance live price feeds. If a stop-loss or profit target is
hit intraday, the position is exited immediately and an alert is sent.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import yfinance as yf

sys.path.append(str(Path(__file__).resolve().parent))
from configs import config
from src.portfolio.manager import _load_state, _save_state
from src.utils.telegram_alerts import send_telegram_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("intraday_tracker")


def run_tracker():
    state = _load_state()
    open_positions = state["open_positions"]
    
    if not open_positions:
        logger.info("No active open positions to track.")
        return

    logger.info("Checking %d open positions for intraday triggers...", len(open_positions))
    
    cash = state["cash"]
    closed_trades = state["closed_trades"]
    updated_open_positions = {}
    state_changed = False
    
    # Process each open position symbol-by-symbol for robustness
    for symbol, pos in open_positions.items():
        ticker = f"{symbol}.NS"
        logger.info("Fetching price for %s...", ticker)
        
        try:
            # Fetch latest 1-day history on a 5-minute interval
            ticker_data = yf.Ticker(ticker).history(period="1d", interval="5m")
            if ticker_data.empty:
                logger.warning("No price data returned for %s, skipping.", ticker)
                updated_open_positions[symbol] = pos
                continue
                
            latest_row = ticker_data.iloc[-1]
            low_px = float(latest_row["Low"])
            high_px = float(latest_row["High"])
            open_px = float(latest_row["Open"])
            close_px = float(latest_row["Close"])
            
            stop_price = pos["stop_price"]
            target_price = pos["target_price"]
            
            is_stop_hit = (low_px <= stop_price)
            is_target_hit = (high_px >= target_price)
            
            # Exit evaluation
            if is_stop_hit or is_target_hit:
                state_changed = True
                exit_reason = "stop_hit"
                exit_price = stop_price
                
                # Check for gap down below stop
                if is_stop_hit and open_px < stop_price:
                    exit_price = open_px  # filled at open due to gap-down slippage
                
                if is_stop_hit and is_target_hit:
                    # If both hit in same bar, be conservative and assume stop was hit
                    exit_reason = "stop_hit"
                elif is_target_hit:
                    exit_reason = "profit_target"
                    exit_price = target_price
                    # Check for gap up past target
                    if open_px > target_price:
                        exit_price = open_px  # filled at open due to gap-up
                
                # Calculate exit transaction proceeds
                shares = pos["shares"]
                gross_value = shares * exit_price
                
                brokerage = gross_value * config.BACKTEST_BROKERAGE_PCT_PER_SIDE
                slippage = gross_value * config.BACKTEST_SLIPPAGE_PCT_PER_SIDE
                net_proceeds = gross_value - brokerage - slippage
                
                cash += net_proceeds
                net_pnl = net_proceeds - pos["notional"]
                net_return_pct = (net_pnl / pos["notional"]) * 100
                
                # Retrieve holding days up to execution date (estimated)
                exec_date = str(pd.Timestamp.now().date())
                entry_date_ts = pd.Timestamp(pos["entry_date"])
                holding_days = max(1, (pd.Timestamp.now().normalize() - entry_date_ts).days)
                
                trade_record = {
                    "symbol": symbol,
                    "entry_date": pos["entry_date"],
                    "entry_price": pos["entry_price"],
                    "exit_date": exec_date,
                    "exit_price": exit_price,
                    "shares": shares,
                    "notional_entry": pos["notional"],
                    "notional_exit": net_proceeds,
                    "net_pnl": net_pnl,
                    "net_return_pct": net_return_pct,
                    "exit_reason": exit_reason,
                    "holding_days": holding_days,
                    "cap_bucket": pos.get("cap_bucket", "UNKNOWN"),
                }
                closed_trades.append(trade_record)
                
                # 1. Console Log
                logger.info("--> TRIGGERED EXIT: %s exited via %s at INR %.2f (PnL: INR %.2f / %.2f%%)",
                            symbol, exit_reason, exit_price, net_pnl, net_return_pct)
                
                # 2. Telegram Alert
                emoji = "🚨 STOP LOSS" if exit_reason == "stop_hit" else "🎉 PROFIT TARGET"
                msg = (
                    f"<b>{emoji} TRIGGERED (Intraday)</b>\n"
                    f"Stock: <b>{symbol}</b>\n"
                    f"Exit Price: <code>{exit_price:.2f}</code> (Reason: <i>{exit_reason}</i>)\n"
                    f"Net PnL: <b>{net_pnl:+.2f}</b> (<b>{net_return_pct:+.2f}%</b>)\n"
                    f"Holding: <code>{holding_days} days</code>"
                )
                send_telegram_message(msg)
            else:
                # Keep position open
                logger.info("Position %s is open. Price: %.2f (Stop: %.2f, Target: %.2f)",
                            symbol, close_px, stop_price, target_price)
                updated_open_positions[symbol] = pos
                
        except Exception as exc:
            logger.error("Error checking intraday price for %s: %s", symbol, exc)
            updated_open_positions[symbol] = pos

    if state_changed:
        state["cash"] = cash
        state["open_positions"] = updated_open_positions
        state["closed_trades"] = closed_trades
        _save_state(state)
        logger.info("Portfolio state updated successfully.")


if __name__ == "__main__":
    run_tracker()
