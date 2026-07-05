"""
Phase 7 / metrics — the numbers that actually answer "if I'd traded
this for N years, how much would I have made, and how painful would
the ride have been."
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config


def compute_backtest_metrics(trade_log: pd.DataFrame, equity_curve: pd.DataFrame) -> dict:
    """
    trade_log: one row per CLOSED trade with at least
        ['net_pnl', 'net_return_pct', 'holding_days']
    equity_curve: ['date', 'equity'] sorted ascending, one row per
        trading day the simulation stepped through.
    """
    if trade_log.empty:
        return {"error": "no closed trades — nothing to evaluate"}

    wins = trade_log[trade_log["net_pnl"] > 0]
    losses = trade_log[trade_log["net_pnl"] <= 0]

    win_rate = len(wins) / len(trade_log)
    gross_profit = wins["net_pnl"].sum()
    gross_loss = -losses["net_pnl"].sum()  # positive number
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    expectancy_pct = trade_log["net_return_pct"].mean()
    avg_holding_days = trade_log["holding_days"].mean()

    # --- Equity-curve-based metrics ---
    equity_curve = equity_curve.sort_values("date").reset_index(drop=True)
    equity = equity_curve["equity"].to_numpy()

    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown_pct = drawdown.min() * 100  # negative number

    daily_returns = pd.Series(equity).pct_change().dropna()
    if daily_returns.std() > 0:
        daily_rf = config.RISK_FREE_RATE_ANNUAL / config.TRADING_DAYS_PER_YEAR
        sharpe = (
            (daily_returns.mean() - daily_rf) / daily_returns.std()
            * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        )
    else:
        sharpe = float("nan")

    n_days = len(equity_curve)
    n_years = n_days / config.TRADING_DAYS_PER_YEAR
    if n_years > 0 and equity[0] > 0 and equity[-1] > 0:
        cagr_pct = ((equity[-1] / equity[0]) ** (1 / n_years) - 1) * 100
    else:
        cagr_pct = float("nan")

    total_return_pct = (equity[-1] / equity[0] - 1) * 100 if equity[0] > 0 else float("nan")

    return {
        "n_trades": len(trade_log),
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 2) if np.isfinite(profit_factor) else None,
        "expectancy_pct_per_trade": round(expectancy_pct, 2),
        "avg_holding_days": round(avg_holding_days, 1),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "sharpe_ratio": round(sharpe, 2) if not np.isnan(sharpe) else None,
        "cagr_pct": round(cagr_pct, 2) if not np.isnan(cagr_pct) else None,
        "total_return_pct": round(total_return_pct, 2),
        "starting_equity": round(float(equity[0]), 2),
        "ending_equity": round(float(equity[-1]), 2),
        "n_trading_days_simulated": n_days,
        "gross_profit": round(float(gross_profit), 2),
        "gross_loss": round(float(gross_loss), 2),
    }
