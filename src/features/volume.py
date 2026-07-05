"""
Volume features — OBV, Chaikin Money Flow, rolling VWAP, volume ratio.

Note: true VWAP is an intraday concept (cumulative within one session).
Since bhavcopy only gives us daily bars, we compute a rolling N-day
VWAP as a proxy for "volume-weighted average price over the recent
trading range" — still useful as a mean-reversion / support reference,
just not the same thing as intraday VWAP.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    # --- On-Balance Volume ---
    direction = np.sign(close.diff()).fillna(0)
    df["obv"] = (direction * volume).cumsum()
    # Raw OBV isn't comparable across stocks of different sizes; a
    # normalized slope is more useful as an ML feature.
    df["obv_slope_10"] = df["obv"].diff(10)

    # --- Chaikin Money Flow ---
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    money_flow_volume = money_flow_multiplier * volume
    df["cmf_20"] = (
        money_flow_volume.rolling(config.CMF_PERIOD).sum()
        / volume.rolling(config.CMF_PERIOD).sum()
    )

    # --- Rolling VWAP (proxy, see module docstring) ---
    typical_price = (high + low + close) / 3
    tp_vol = typical_price * volume
    rolling_vwap = (
        tp_vol.rolling(config.VWAP_PERIOD).sum()
        / volume.rolling(config.VWAP_PERIOD).sum()
    )
    df["vwap_20"] = rolling_vwap
    df["dist_from_vwap_pct"] = (close - rolling_vwap) / rolling_vwap * 100

    # --- Volume ratio: today's volume vs recent average (spike detector) ---
    avg_volume = volume.rolling(config.VOLUME_RATIO_PERIOD).mean()
    df["volume_ratio"] = volume / avg_volume.replace(0, np.nan)

    return df
