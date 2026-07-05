"""
Portfolio Manager for Paper Trading

Manages cash, open positions, pending orders, and transaction history.
Executes recommended scanner signals at the next day's open price (to prevent
leakage) and checks open positions daily against actual price limits.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config
from src.risk.dynamic_exit import simulate_dynamic_exit
from src.risk.position_sizing import position_size_by_risk, compute_atr_stop_price

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("portfolio_manager")


def _load_state() -> dict:
    """Load the persistent portfolio state, initializing if necessary."""
    if not config.PAPER_PORTFOLIO_FILE.exists():
        reset_portfolio()
    with open(config.PAPER_PORTFOLIO_FILE) as f:
        return json.load(f)


def _save_state(state: dict) -> None:
    """Save the portfolio state to disk atomically to prevent corruption."""
    temp_file = config.PAPER_PORTFOLIO_FILE.with_suffix(".tmp")
    with open(temp_file, "w") as f:
        json.dump(state, f, indent=2)
    # Atomic replace
    temp_file.replace(config.PAPER_PORTFOLIO_FILE)


def reset_portfolio(initial_cash: float = None) -> dict:
    """Reset the portfolio to initial cash balance and clear all trades."""
    initial_cash = initial_cash or config.PAPER_STARTING_CASH
    state = {
        "cash": initial_cash,
        "open_positions": {},
        "pending_entries": [],
        "closed_trades": [],
    }
    config.PAPER_PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
    _save_state(state)

    # Initialize equity history file
    if config.PAPER_EQUITY_FILE.exists():
        config.PAPER_EQUITY_FILE.unlink()
    
    with open(config.PAPER_EQUITY_FILE, "w") as f:
        f.write("date,cash,positions_value,total_equity\n")

    logger.info("Portfolio reset to starting cash: INR %.2f", initial_cash)
    return state


def get_portfolio_summary() -> dict:
    """Compute total equity and return structured performance summary."""
    state = _load_state()
    cash = state["cash"]
    open_positions = state["open_positions"]
    
    # Calculate current market value of open positions.
    positions_value = 0.0
    active_positions_details = []
    
    for symbol, pos in open_positions.items():
        ohlcv_path = config.OHLCV_DIR / f"{symbol}.csv"
        current_price = pos["entry_price"]
        if ohlcv_path.exists():
            df = pd.read_csv(ohlcv_path)
            if not df.empty:
                current_price = float(df.sort_values("date").iloc[-1]["close"])
        
        current_value = pos["shares"] * current_price
        positions_value += current_value
        
        pnl = current_value - pos["notional"]
        pnl_pct = (pnl / pos["notional"]) * 100
        
        active_positions_details.append({
            "symbol": symbol,
            "entry_date": pos["entry_date"],
            "entry_price": pos["entry_price"],
            "current_price": current_price,
            "shares": pos["shares"],
            "notional": pos["notional"],
            "current_value": current_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "stop_price": pos["stop_price"],
            "target_price": pos["target_price"],
            "cap_bucket": pos.get("cap_bucket", "UNKNOWN"),
        })

    total_equity = cash + positions_value
    
    # Compute trade metrics.
    trades = pd.DataFrame(state["closed_trades"])
    metrics = {}
    if not trades.empty:
        wins = trades[trades["net_pnl"] > 0]
        losses = trades[trades["net_pnl"] <= 0]
        metrics["n_trades"] = len(trades)
        metrics["win_rate_pct"] = round(len(wins) / len(trades) * 100, 2)
        metrics["total_net_pnl"] = round(trades["net_pnl"].sum(), 2)
        metrics["profit_factor"] = round(wins["net_pnl"].sum() / abs(losses["net_pnl"].sum()), 2) if not losses.empty else float("inf")
        metrics["expectancy_pct"] = round(trades["net_return_pct"].mean(), 2)
        metrics["avg_holding_days"] = round(trades["holding_days"].mean(), 1)
    else:
        metrics["n_trades"] = 0
        metrics["win_rate_pct"] = 0.0
        metrics["total_net_pnl"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["expectancy_pct"] = 0.0
        metrics["avg_holding_days"] = 0.0

    return {
        "cash": cash,
        "positions_value": positions_value,
        "total_equity": total_equity,
        "open_positions": active_positions_details,
        "pending_entries": state["pending_entries"],
        "metrics": metrics,
        "n_closed_trades": len(state["closed_trades"]),
    }


def update_portfolio(current_date: str) -> None:
    """
    Updates active exits and executes pending buys based on price data up to current_date.
    """
    state = _load_state()
    cash = state["cash"]
    open_positions = state["open_positions"]
    pending_entries = state["pending_entries"]
    closed_trades = state["closed_trades"]
    
    current_date_ts = pd.Timestamp(current_date)
    updated_open_positions = {}
    
    # 1. Update active open positions
    logger.info("Evaluating active open positions for exits up to %s...", current_date)
    for symbol, pos in open_positions.items():
        ohlcv_path = config.OHLCV_DIR / f"{symbol}.csv"
        if not ohlcv_path.exists():
            # Keep open if data file is missing
            updated_open_positions[symbol] = pos
            continue
            
        df = pd.read_csv(ohlcv_path, parse_dates=["date"]).set_index("date").sort_index()
        # Find price data since day after entry
        entry_date_ts = pd.Timestamp(pos["entry_date"])
        post_entry_df = df[df.index > entry_date_ts]
        
        if post_entry_df.empty:
            updated_open_positions[symbol] = pos
            continue
            
        # Run dynamic exit simulation
        # Using configured trailing stop or static parameters based on state
        exit_result = simulate_dynamic_exit(
            ohlcv=df,
            entry_date=entry_date_ts,
            entry_price=pos["entry_price"],
            atr_pct_at_signal=pos["atr_pct"],
        )
        
        if exit_result is not None and pd.Timestamp(exit_result["exit_date"]) <= current_date_ts:
            # Position exited!
            exit_price = float(exit_result["exit_price"])
            exit_date = str(pd.Timestamp(exit_result["exit_date"]).date())
            holding_days = int(exit_result["holding_days"])
            reason = exit_result["exit_reason"]
            
            gross_value = pos["shares"] * exit_price
            brokerage = gross_value * config.BACKTEST_BROKERAGE_PCT_PER_SIDE
            slippage = gross_value * config.BACKTEST_SLIPPAGE_PCT_PER_SIDE
            net_proceeds = gross_value - brokerage - slippage
            
            cash += net_proceeds
            net_pnl = net_proceeds - pos["notional"]
            net_return_pct = (net_pnl / pos["notional"]) * 100
            
            trade_record = {
                "symbol": symbol,
                "entry_date": pos["entry_date"],
                "entry_price": pos["entry_price"],
                "exit_date": exit_date,
                "exit_price": exit_price,
                "shares": pos["shares"],
                "notional_entry": pos["notional"],
                "notional_exit": net_proceeds,
                "net_pnl": net_pnl,
                "net_return_pct": net_return_pct,
                "exit_reason": reason,
                "holding_days": holding_days,
                "cap_bucket": pos.get("cap_bucket", "UNKNOWN"),
            }
            closed_trades.append(trade_record)
            logger.info("--> CLOSED position in %s on %s at INR %.2f (%s, net PnL: INR %.2f / %.2f%%)", 
                        symbol, exit_date, exit_price, reason, net_pnl, net_return_pct)
        else:
            # Position remains open
            updated_open_positions[symbol] = pos

    # 2. Execute pending buys
    logger.info("Processing pending entries...")
    remaining_pending = []
    
    # Determine market regime for adaptive risk scaling
    from src.regime.regime_detector import classify_current_regime
    regime_info = classify_current_regime()
    regime = regime_info["regime"]
    regime_mult = config.REGIME_RISK_MULTIPLIERS.get(regime, 1.0)
    logger.info("Market regime at execution: %s (Multiplier: %.2f)", regime, regime_mult)

    total_positions_value = sum(pos["shares"] * pos["entry_price"] for pos in updated_open_positions.values())
    current_equity = cash + total_positions_value

    for entry in pending_entries:
        symbol = entry["symbol"]
        
        # Prevent buying if already in open_positions
        if symbol in updated_open_positions:
            continue
            
        # If Bear market (multiplier is 0), skip execution entirely
        if regime_mult == 0.0:
            logger.info("Skipped pending signal %s due to BEAR market regime deactivation filter", symbol)
            continue

        ohlcv_path = config.OHLCV_DIR / f"{symbol}.csv"
        if not ohlcv_path.exists():
            remaining_pending.append(entry)
            continue
            
        df = pd.read_csv(ohlcv_path, parse_dates=["date"]).set_index("date").sort_index()
        signal_date_ts = pd.Timestamp(entry["signal_date"])
        
        # Look for dates after signal date
        post_signal_df = df[df.index > signal_date_ts]
        if post_signal_df.empty:
            remaining_pending.append(entry)
            continue
            
        # Get next day's open price
        exec_row = post_signal_df.iloc[0]
        exec_date = str(post_signal_df.index[0].date())
        open_price = float(exec_row["open"])
        
        # Size position using dynamic risk allocation scaled by regime multiplier
        stop_price = compute_atr_stop_price(open_price, entry["atr_pct"])
        sizing = position_size_by_risk(
            current_equity=current_equity,
            entry_price=open_price,
            stop_price=stop_price,
            risk_pct=config.RISK_PER_TRADE_PCT * regime_mult,
        )
        
        shares = sizing["shares"]
        notional_cost = sizing["notional"]
        
        brokerage = notional_cost * config.BACKTEST_BROKERAGE_PCT_PER_SIDE
        slippage = notional_cost * config.BACKTEST_SLIPPAGE_PCT_PER_SIDE
        total_cost = notional_cost + brokerage + slippage
        
        if total_cost <= cash and shares > 0:
            cash -= total_cost
            
            # Setup active position details
            updated_open_positions[symbol] = {
                "symbol": symbol,
                "entry_date": exec_date,
                "entry_price": open_price,
                "shares": shares,
                "notional": notional_cost,
                "stop_price": stop_price,
                "target_price": open_price * (1 + config.TRIPLE_BARRIER_PROFIT_PCT),
                "atr_pct": entry["atr_pct"],
                "cap_bucket": entry["cap_bucket"],
            }
            logger.info("<-- OPENED position in %s on %s at INR %.2f (Shares: %.2f, Cost: INR %.2f)",
                        symbol, exec_date, open_price, shares, total_cost)
        else:
            logger.info("Skipped executing pending signal %s (insufficient cash or invalid sizing)", symbol)

    # 3. Save updated state back
    state["cash"] = cash
    state["open_positions"] = updated_open_positions
    state["pending_entries"] = remaining_pending
    state["closed_trades"] = closed_trades
    _save_state(state)


def add_pending_entries(scanner_report: pd.DataFrame, current_date: str) -> None:
    """
    Appends recommended scan signals to pending_entries.
    """
    state = _load_state()
    open_positions = state["open_positions"]
    pending_entries = state["pending_entries"]
    
    # Filter scanner signals above 50% probability and not already held
    candidates = scanner_report[
        (scanner_report["probability"] >= 0.50) & 
        (~scanner_report["symbol"].isin(open_positions)) &
        (~scanner_report["symbol"].isin([p["symbol"] for p in pending_entries]))
    ]
    
    # Retrieve details to store in pending
    new_pendings = []
    for _, row in candidates.iterrows():
        # Get ATR pct from signal features
        symbol = row["symbol"]
        features_path = config.FEATURES_DIR / f"{symbol}.csv"
        atr_pct = config.TRIPLE_BARRIER_STOP_PCT * 100 / config.ATR_STOP_MULTIPLIER # fallback
        if features_path.exists():
            feat_df = pd.read_csv(features_path)
            if not feat_df.empty:
                atr_pct = float(feat_df.sort_values("date").iloc[-1]["atr_pct"])

        new_pendings.append({
            "symbol": symbol,
            "signal_date": current_date,
            "probability": float(row["probability"]),
            "atr_pct": atr_pct,
            "cap_bucket": row["cap_bucket"],
        })
        
    # Standard queue limit to avoid stacking too many signals if cash is low
    pending_entries.extend(new_pendings)
    state["pending_entries"] = pending_entries
    _save_state(state)
    logger.info("Added %d new candidates to pending queue for next trading session execution.", len(new_pendings))


def log_daily_equity(current_date: str) -> None:
    """
    Calculates current portfolio value and appends it to configs.PAPER_EQUITY_FILE.
    """
    summary = get_portfolio_summary()
    cash = summary["cash"]
    positions_value = summary["positions_value"]
    total_equity = summary["total_equity"]
    
    # Check if date already logged to prevent duplicates
    if config.PAPER_EQUITY_FILE.exists():
        df = pd.read_csv(config.PAPER_EQUITY_FILE)
        if current_date in df["date"].astype(str).values:
            return
            
    with open(config.PAPER_EQUITY_FILE, "a") as f:
        f.write(f"{current_date},{cash:.2f},{positions_value:.2f},{total_equity:.2f}\n")
    logger.info("Equity logged for %s: INR %.2f (Cash: INR %.2f, Open positions: INR %.2f)",
                current_date, total_equity, cash, positions_value)
