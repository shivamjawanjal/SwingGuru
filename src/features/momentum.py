"""
Momentum features — RSI, MACD, ROC, plain Momentum.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing = an EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # When avg_loss is 0 and avg_gain > 0, RSI should be 100.
    rsi = rsi.where(avg_loss != 0, 100)
    return rsi


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    # --- RSI ---
    df[f"rsi_{config.RSI_PERIOD}"] = _rsi(close, config.RSI_PERIOD)

    # --- MACD ---
    ema_fast = close.ewm(span=config.MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=config.MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=config.MACD_SIGNAL, adjust=False).mean()
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd_line - signal_line
    df["macd_bullish_cross"] = (
        (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    ).astype(int)
    df["macd_bearish_cross"] = (
        (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    ).astype(int)

    # --- Rate of Change ---
    for period in config.ROC_PERIODS:
        df[f"roc_{period}"] = (close - close.shift(period)) / close.shift(period) * 100

    # --- Plain Momentum ---
    for period in config.MOMENTUM_PERIODS:
        df[f"momentum_{period}"] = close - close.shift(period)

    return df
