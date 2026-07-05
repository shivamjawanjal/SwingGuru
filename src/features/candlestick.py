"""
Candlestick pattern features — Doji, Hammer, Inside Bar, Engulfing.

All patterns are computed vectorized (no row-by-row loops) and emit
0/1 flag columns so they drop straight into the ML feature matrix.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config


def add_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    body = (c - o).abs()
    day_range = (h - l).replace(0, np.nan)
    upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l
    is_bullish = c > o
    is_bearish = c < o

    # --- Doji: body is a tiny fraction of the day's range ---
    df["doji"] = (body <= config.DOJI_BODY_RATIO * day_range).astype(int)

    # --- Inside Bar: today's range is fully inside yesterday's range ---
    df["inside_bar"] = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)

    # --- Hammer: long lower shadow, small upper shadow, body near the top ---
    df["hammer"] = (
        (lower_shadow >= config.HAMMER_LOWER_SHADOW_MULT * body.replace(0, np.nan))
        & (upper_shadow <= config.HAMMER_UPPER_SHADOW_MAX_RATIO * day_range)
        & (body > 0)
    ).fillna(False).astype(int)

    # --- Engulfing: today's real body fully engulfs yesterday's real body,
    #     with an opposite-direction flip ---
    prev_open, prev_close = o.shift(1), c.shift(1)
    prev_bearish = prev_close < prev_open
    prev_bullish = prev_close > prev_open

    df["bullish_engulfing"] = (
        is_bullish & prev_bearish & (o <= prev_close) & (c >= prev_open)
    ).astype(int)
    df["bearish_engulfing"] = (
        is_bearish & prev_bullish & (o >= prev_close) & (c <= prev_open)
    ).astype(int)

    return df
