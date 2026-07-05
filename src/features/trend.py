"""
Trend features — EMAs and price-vs-EMA distances.

All functions take a DataFrame sorted by date ascending with at least
a 'close' column, and return the SAME dataframe with new columns added
(never mutate in place — callers chain these together).
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config


def add_ema_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for period in config.EMA_PERIODS:
        col = f"ema_{period}"
        df[col] = df["close"].ewm(span=period, adjust=False).mean()
        # Distance from EMA as a %, one of the strongest simple trend features.
        df[f"dist_from_{col}_pct"] = (df["close"] - df[col]) / df[col] * 100

    # Trend alignment: are shorter EMAs above longer EMAs? (classic
    # "stacked EMA" bullish/bearish structure)
    df["ema_20_above_50"] = (df["ema_20"] > df["ema_50"]).astype(int)
    df["ema_50_above_100"] = (df["ema_50"] > df["ema_100"]).astype(int)
    df["ema_100_above_200"] = (df["ema_100"] > df["ema_200"]).astype(int)
    df["ema_bullish_stack"] = (
        df["ema_20_above_50"] & df["ema_50_above_100"] & df["ema_100_above_200"]
    ).astype(int)

    return df
