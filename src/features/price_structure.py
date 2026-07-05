"""
Price structure features — swing structure, gaps, breakouts, support/resistance.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config


def add_price_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]
    n = config.PRICE_STRUCTURE_LOOKBACK

    # --- Day-over-day swing structure ---
    df["higher_high"] = (high > high.shift(1)).astype(int)
    df["higher_low"] = (low > low.shift(1)).astype(int)
    df["lower_high"] = (high < high.shift(1)).astype(int)
    df["lower_low"] = (low < low.shift(1)).astype(int)
    # "Healthy uptrend day" = both higher high AND higher low
    df["uptrend_structure"] = (df["higher_high"] & df["higher_low"]).astype(int)
    df["downtrend_structure"] = (df["lower_high"] & df["lower_low"]).astype(int)

    # --- Gaps ---
    prev_close = close.shift(1)
    gap_pct = (open_ - prev_close) / prev_close
    df["gap_pct"] = gap_pct * 100
    df["gap_up"] = (gap_pct >= config.GAP_THRESHOLD_PCT).astype(int)
    df["gap_down"] = (gap_pct <= -config.GAP_THRESHOLD_PCT).astype(int)

    # --- Support / Resistance (rolling N-day extremes, shifted so
    #     "today" isn't included in its own reference level) ---
    resistance = high.rolling(n).max().shift(1)
    support = low.rolling(n).min().shift(1)
    df[f"resistance_{n}d"] = resistance
    df[f"support_{n}d"] = support
    df["dist_from_resistance_pct"] = (close - resistance) / resistance * 100
    df["dist_from_support_pct"] = (close - support) / support * 100

    # --- Breakouts: closing beyond the prior N-day range ---
    df["breakout_up"] = (close > resistance).astype(int)
    df["breakout_down"] = (close < support).astype(int)

    return df
