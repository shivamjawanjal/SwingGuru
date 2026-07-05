"""
Triple-barrier labeling.

For every candidate entry day, look forward up to `max_days` trading
days and ask: does the price hit the profit target before the stop
loss?

  - profit target hit first -> label = 1 (positive trade)
  - stop loss hit first     -> label = 0 (negative trade)
  - neither hit by the end of the window (time-barrier / vertical
    barrier) -> label = 0 (the doc's spec is binary: "if yes positive
    trade, otherwise negative trade" — a trade that never reached
    target within the holding window is not a swing-trade win, even
    if it technically ended up slightly positive)

Entry is priced at the open `ENTRY_LAG_DAYS` trading days after the
signal day — NOT the signal day's own close — to avoid look-ahead bias.
A model trained on "today's close" as the effective entry price would
be learning something you can't actually replicate live.

This module operates on numpy arrays (not pandas row-apply) for speed:
each entry point requires a small forward scan (<= max_days), and doing
that scan in vectorized/loop-of-arrays form rather than DataFrame.apply
keeps a full 6-year single-symbol run in the tens-of-milliseconds range.
"""

from typing import Optional

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config


def _resolve_same_day_tie(day_open: float, upper: float, lower: float) -> str:
    """
    Both barriers touched within the same day's high/low range.
    We don't have intraday sequencing, so approximate: whichever level
    is numerically closer to the day's open is assumed reached first
    (price moves continuously outward from the open).
    """
    dist_to_upper = abs(upper - day_open)
    dist_to_lower = abs(day_open - lower)
    return "profit_target" if dist_to_upper <= dist_to_lower else "stop_loss"


def triple_barrier_labels(
    df: pd.DataFrame,
    profit_pct: float = None,
    stop_pct: float = None,
    max_days: int = None,
    entry_lag_days: int = None,
) -> pd.DataFrame:
    """
    df must be sorted ascending by date with columns: date, open, high, low, close.
    Returns a NEW dataframe with one row per original row plus label columns:
      entry_date, entry_price, exit_date, exit_price, exit_reason,
      holding_days, realized_return_pct, label
    Rows too close to the end of the series to have a full lookahead
    window get label = NaN (drop these before training — they're
    censored, not negative).
    """
    profit_pct = profit_pct if profit_pct is not None else config.TRIPLE_BARRIER_PROFIT_PCT
    stop_pct = stop_pct if stop_pct is not None else config.TRIPLE_BARRIER_STOP_PCT
    max_days = max_days if max_days is not None else config.TRIPLE_BARRIER_MAX_DAYS
    entry_lag_days = entry_lag_days if entry_lag_days is not None else config.ENTRY_LAG_DAYS

    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)

    dates = df["date"].values
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    entry_date = np.full(n, None, dtype=object)
    entry_price = np.full(n, np.nan)
    exit_date = np.full(n, None, dtype=object)
    exit_price = np.full(n, np.nan)
    exit_reason = np.full(n, None, dtype=object)
    holding_days = np.full(n, np.nan)
    realized_return_pct = np.full(n, np.nan)
    label = np.full(n, np.nan)

    for i in range(n):
        entry_idx = i + entry_lag_days
        if entry_idx >= n:
            continue  # not enough future data to even enter

        e_price = opens[entry_idx]
        upper = e_price * (1 + profit_pct)
        lower = e_price * (1 - stop_pct)

        # The window is max_days trading days long, starting at entry_idx.
        full_window_end = entry_idx + max_days - 1
        window_end = min(full_window_end, n - 1)
        window_is_complete = full_window_end <= n - 1
        resolved = False

        for day_idx in range(entry_idx, window_end + 1):
            hit_upper = highs[day_idx] >= upper
            hit_lower = lows[day_idx] <= lower

            if hit_upper and hit_lower:
                reason = _resolve_same_day_tie(opens[day_idx], upper, lower)
            elif hit_upper:
                reason = "profit_target"
            elif hit_lower:
                reason = "stop_loss"
            else:
                continue

            entry_date[i] = dates[entry_idx]
            entry_price[i] = e_price
            exit_date[i] = dates[day_idx]
            exit_price[i] = upper if reason == "profit_target" else lower
            exit_reason[i] = reason
            holding_days[i] = day_idx - entry_idx + 1
            realized_return_pct[i] = (exit_price[i] / e_price - 1) * 100
            label[i] = 1.0 if reason == "profit_target" else 0.0
            resolved = True
            break

        if resolved:
            continue

        if window_is_complete:
            # Neither barrier touched across the full max_days window ->
            # genuine timeout, exits at the last day's close.
            last_idx = window_end
            entry_date[i] = dates[entry_idx]
            entry_price[i] = e_price
            exit_date[i] = dates[last_idx]
            exit_price[i] = closes[last_idx]
            exit_reason[i] = "timeout"
            holding_days[i] = last_idx - entry_idx + 1
            realized_return_pct[i] = (closes[last_idx] / e_price - 1) * 100
            label[i] = 0.0
        # else: window was cut short by the end of available data and no
        # barrier was touched in what we could see — outcome is genuinely
        # unknown (censored). Leave as NaN; these rows get dropped before
        # training rather than mislabeled.

    out = df.copy()
    out["entry_date"] = entry_date
    out["entry_price"] = entry_price
    out["exit_date"] = exit_date
    out["exit_price"] = exit_price
    out["exit_reason"] = exit_reason
    out["holding_days"] = holding_days
    out["realized_return_pct"] = realized_return_pct
    out["label"] = label

    return out
