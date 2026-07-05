"""
Phase 9 / position_sizing — size positions by RISK, not by an equal
1/N split of capital.

Phase 7's backtest gave every position the same notional amount
(current_equity / max_positions) regardless of how far away its stop
was. That means a low-volatility stock with a tight stop and a
high-volatility stock with a wide stop got the exact same dollar
exposure, which means very different dollar RISK — the whole point of
risk management is to fix that.

Here, position size is chosen so that if the stop is hit, the loss
equals a fixed fraction of current equity (config.RISK_PER_TRADE_PCT),
using each stock's own ATR (already computed in Phase 2, as of the
signal day — no lookahead) as the stop-distance reference instead of
one fixed percentage for every stock.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config


def compute_atr_stop_price(entry_price: float, atr_pct_at_signal: float, atr_multiplier: Optional[float] = None) -> float:
    """
    Initial stop-loss price = entry - ATR_STOP_MULTIPLIER * ATR.
    atr_pct_at_signal is Phase 2's atr_pct feature (ATR14 as a % of
    close), so the stop distance scales with each stock's own recent
    volatility instead of a single fixed % for every stock.
    """
    multiplier = atr_multiplier if atr_multiplier is not None else config.ATR_STOP_MULTIPLIER
    stop_distance_pct = multiplier * (atr_pct_at_signal / 100)
    return entry_price * (1 - stop_distance_pct)


def position_size_by_risk(
    current_equity: float,
    entry_price: float,
    stop_price: float,
    risk_pct: Optional[float] = None,
    max_notional_pct: Optional[float] = None,
) -> dict:
    """
    Returns {'shares', 'notional', 'risk_amount', 'capped_by_max_notional'}.

    Base case: shares sized so (entry_price - stop_price) * shares equals
    risk_pct of current_equity — if the stop is hit, you lose exactly
    that fraction of equity regardless of how volatile the stock is or
    how far away its stop happens to sit.

    Safety cap: a very tight ATR stop on a low-volatility stock could
    otherwise imply putting an enormous fraction of equity into one
    name to hit the target dollar risk. max_notional_pct caps any
    single position's notional regardless of what the risk formula
    alone would suggest — the actual dollar risk taken is then smaller
    than risk_pct, reported honestly in risk_amount.
    """
    risk_pct = risk_pct if risk_pct is not None else config.RISK_PER_TRADE_PCT
    max_notional_pct = max_notional_pct if max_notional_pct is not None else config.MAX_POSITION_NOTIONAL_PCT_OF_EQUITY

    stop_distance_per_share = entry_price - stop_price
    if stop_distance_per_share <= 0 or entry_price <= 0:
        return {"shares": 0.0, "notional": 0.0, "risk_amount": 0.0, "capped_by_max_notional": False}

    risk_amount = current_equity * risk_pct
    shares = risk_amount / stop_distance_per_share
    notional = shares * entry_price

    max_notional = current_equity * max_notional_pct
    capped = notional > max_notional
    if capped:
        notional = max_notional
        shares = notional / entry_price
        risk_amount = shares * stop_distance_per_share  # actual risk taken, honestly smaller

    return {
        "shares": shares,
        "notional": notional,
        "risk_amount": risk_amount,
        "capped_by_max_notional": capped,
    }
