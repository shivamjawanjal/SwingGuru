"""
Volatility features — ATR, Bollinger Band width, historical volatility.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- ATR (Wilder's smoothing, same alpha=1/period convention as RSI) ---
    tr = _true_range(df)
    df["atr_14"] = tr.ewm(alpha=1 / config.ATR_PERIOD, min_periods=config.ATR_PERIOD, adjust=False).mean()
    df["atr_pct"] = df["atr_14"] / df["close"] * 100  # normalized, comparable across stocks

    # --- Bollinger Bands ---
    mid = df["close"].rolling(config.BOLLINGER_PERIOD).mean()
    std = df["close"].rolling(config.BOLLINGER_PERIOD).std()
    upper = mid + config.BOLLINGER_STD * std
    lower = mid - config.BOLLINGER_STD * std
    df["bollinger_width_pct"] = (upper - lower) / mid * 100
    df["bollinger_pct_b"] = (df["close"] - lower) / (upper - lower).replace(0, np.nan)

    # --- Historical Volatility (annualized stdev of log returns) ---
    log_returns = np.log(df["close"] / df["close"].shift(1))
    df["hist_volatility_annualized_pct"] = (
        log_returns.rolling(config.HIST_VOL_PERIOD).std()
        * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        * 100
    )

    return df
