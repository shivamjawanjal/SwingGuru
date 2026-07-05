"""
Phase 9 / dynamic_exit — simulate a trade's actual exit using an
ATR-based stop that trails upward as price moves favorably, instead of
Phase 3's static triple-barrier exit (fixed +8%/-4%, decided once at
label time).

This matters because a fixed % stop ignores how volatile the specific
stock is, and a static profit/stop barrier can't lock in gains on a
trade that ran up 15% before pulling back to +8% — a trailing stop
would have exited near the peak instead of riding it all the way back
down to the original target.

Walks forward day by day from the entry date using that symbol's own
daily OHLCV (not the pre-computed label), so this is a genuinely
different simulation of "what would have happened," not just a
relabeling of Phase 3's output.
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config


def _resolve_same_day_tie(day_open: float, stop_price: float, target_price: float) -> str:
    """Same convention as Phase 3: whichever level is closer to the day's open is assumed hit first."""
    dist_to_target = abs(target_price - day_open)
    dist_to_stop = abs(day_open - stop_price)
    return "profit_target" if dist_to_target <= dist_to_stop else "stop_hit"


def simulate_dynamic_exit(
    ohlcv: pd.DataFrame,
    entry_date: pd.Timestamp,
    entry_price: float,
    atr_pct_at_signal: float,
    profit_pct: Optional[float] = None,
    max_days: Optional[int] = None,
    atr_stop_multiplier: Optional[float] = None,
    trailing_stop_atr_multiplier: Optional[float] = None,
    enable_trailing_stop: Optional[bool] = None,
) -> Optional[dict]:
    """
    ohlcv: full daily OHLCV for ONE symbol, indexed by date ascending,
           columns ['open','high','low','close'].
    Returns None if entry_date isn't found or there's no data at/after it.

    Otherwise returns {'exit_date','exit_price','exit_reason',
    'holding_days','final_stop_price','trailing_stop_ever_ratcheted'}.
    exit_reason is one of: 'profit_target', 'initial_atr_stop',
    'trailing_stop', 'timeout'.
    """
    profit_pct = profit_pct if profit_pct is not None else config.TRIPLE_BARRIER_PROFIT_PCT
    max_days = max_days if max_days is not None else config.TRIPLE_BARRIER_MAX_DAYS
    atr_stop_multiplier = atr_stop_multiplier if atr_stop_multiplier is not None else config.ATR_STOP_MULTIPLIER
    trailing_mult = trailing_stop_atr_multiplier if trailing_stop_atr_multiplier is not None else config.TRAILING_STOP_ATR_MULTIPLIER
    enable_trailing = enable_trailing_stop if enable_trailing_stop is not None else config.ENABLE_TRAILING_STOP

    if entry_date not in ohlcv.index:
        return None
    entry_idx = ohlcv.index.get_loc(entry_date)
    n = len(ohlcv)

    atr_absolute = entry_price * (atr_pct_at_signal / 100)
    initial_stop = entry_price * (1 - atr_stop_multiplier * (atr_pct_at_signal / 100))
    target_price = entry_price * (1 + profit_pct)

    running_stop = initial_stop
    highest_close_since_entry = entry_price
    ratcheted = False

    window_end_idx = min(entry_idx + max_days - 1, n - 1)
    window_is_complete = (entry_idx + max_days - 1) <= n - 1

    for day_idx in range(entry_idx, window_end_idx + 1):
        day = ohlcv.iloc[day_idx]

        if enable_trailing and day_idx > entry_idx:
            # Ratchet based on the highest CLOSE seen so far (not
            # intraday high) — closes are what we reliably have for
            # every day; using intraday highs on daily bars overstates
            # how tightly a real trailing stop could actually track.
            if day["close"] > highest_close_since_entry:
                highest_close_since_entry = day["close"]
            trailing_candidate = highest_close_since_entry - atr_absolute * trailing_mult
            if trailing_candidate > running_stop:
                running_stop = trailing_candidate
                ratcheted = True

        hit_target = day["high"] >= target_price
        hit_stop = day["low"] <= running_stop

        if hit_target and hit_stop:
            reason = _resolve_same_day_tie(day["open"], running_stop, target_price)
            exit_reason = "profit_target" if reason == "profit_target" else (
                "trailing_stop" if ratcheted else "initial_atr_stop"
            )
            exit_price = target_price if reason == "profit_target" else running_stop
        elif hit_target:
            exit_reason, exit_price = "profit_target", target_price
        elif hit_stop:
            exit_reason = "trailing_stop" if ratcheted else "initial_atr_stop"
            exit_price = running_stop
        else:
            continue

        return {
            "exit_date": ohlcv.index[day_idx],
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "holding_days": day_idx - entry_idx + 1,
            "final_stop_price": running_stop,
            "trailing_stop_ever_ratcheted": ratcheted,
        }

    if window_is_complete:
        last_idx = window_end_idx
        return {
            "exit_date": ohlcv.index[last_idx],
            "exit_price": ohlcv.iloc[last_idx]["close"],
            "exit_reason": "timeout",
            "holding_days": last_idx - entry_idx + 1,
            "final_stop_price": running_stop,
            "trailing_stop_ever_ratcheted": ratcheted,
        }

    return None  # censored — ran out of data before the window completed
