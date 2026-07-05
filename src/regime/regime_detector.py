"""
Market Regime Detection

Classifies the current (and historical) market environment into one of
four regimes using a proxy index built from large-cap Nifty50 component
stocks already in data/ohlcv/:

  - BULL:      rolling return > threshold, volatility < vol threshold
  - BEAR:      rolling return < -threshold, volatility < vol threshold
  - HIGH_VOL:  volatility >= vol threshold (regardless of return)
  - SIDEWAYS:  everything else

Why a proxy basket instead of Nifty50 index data?
NSE doesn't publish index-level OHLCV in the bhavcopy files. Rather
than adding a separate data source (and a separate failure point), we
compute an equal-weighted basket of 20 large-cap stocks — this tracks
the broad market closely enough for regime classification while reusing
data the pipeline already downloads.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("regime_detector")

# Canonical regime labels.
BULL = "BULL"
BEAR = "BEAR"
HIGH_VOL = "HIGH_VOL"
SIDEWAYS = "SIDEWAYS"


def _build_proxy_index() -> pd.DataFrame:
    """
    Builds a daily equal-weighted proxy index from config.REGIME_PROXY_SYMBOLS.
    Returns a DataFrame with columns [date, proxy_close] sorted by date.
    Uses percentage returns for equal-weighting so stock price scale
    doesn't matter.
    """
    daily_returns = []
    available_symbols = []

    for symbol in config.REGIME_PROXY_SYMBOLS:
        path = config.OHLCV_DIR / f"{symbol}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["date"])[["date", "close"]]
        df = df.sort_values("date").set_index("date")
        ret = df["close"].pct_change()
        daily_returns.append(ret.rename(symbol))
        available_symbols.append(symbol)

    if len(daily_returns) < 5:
        raise RuntimeError(
            f"Only {len(daily_returns)} proxy symbols found in OHLCV data "
            f"(need at least 5). Available: {available_symbols}. "
            "Ensure Phase 1 has been run."
        )

    logger.info("Proxy index built from %d/%d configured symbols: %s",
                len(available_symbols), len(config.REGIME_PROXY_SYMBOLS),
                available_symbols[:5])

    # Equal-weighted daily return = mean across all component returns.
    combined = pd.concat(daily_returns, axis=1)
    proxy_return = combined.mean(axis=1)

    # Reconstruct an index level starting at 100.
    proxy_close = (1 + proxy_return).cumprod() * 100
    proxy_close.iloc[0] = 100.0  # first day has NaN return -> set base

    result = pd.DataFrame({"date": proxy_close.index, "proxy_close": proxy_close.values})
    result = result.dropna(subset=["proxy_close"]).reset_index(drop=True)
    return result


def compute_regime_history(
    return_window: Optional[int] = None,
    vol_window: Optional[int] = None,
    return_threshold: Optional[float] = None,
    vol_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        [date, proxy_close, rolling_return_pct, rolling_vol_annualized_pct, regime]
    One row per trading day.
    """
    return_window = return_window or config.REGIME_RETURN_WINDOW
    vol_window = vol_window or config.REGIME_VOLATILITY_WINDOW
    return_threshold = return_threshold if return_threshold is not None else config.REGIME_RETURN_THRESHOLD_PCT
    vol_threshold = vol_threshold if vol_threshold is not None else config.REGIME_VOLATILITY_THRESHOLD_PCT

    proxy = _build_proxy_index()
    close = proxy["proxy_close"]

    # Rolling return over the window (percentage).
    rolling_return = close.pct_change(periods=return_window) * 100

    # Rolling annualized volatility.
    daily_returns = close.pct_change()
    rolling_vol = daily_returns.rolling(window=vol_window).std() * np.sqrt(config.TRADING_DAYS_PER_YEAR) * 100

    proxy["rolling_return_pct"] = rolling_return.round(2)
    proxy["rolling_vol_annualized_pct"] = rolling_vol.round(2)

    # Classify regime.
    def _classify(row):
        ret = row["rolling_return_pct"]
        vol = row["rolling_vol_annualized_pct"]
        if pd.isna(ret) or pd.isna(vol):
            return None
        if vol >= vol_threshold:
            return HIGH_VOL
        if ret > return_threshold:
            return BULL
        if ret < -return_threshold:
            return BEAR
        return SIDEWAYS

    proxy["regime"] = proxy.apply(_classify, axis=1)
    proxy = proxy.dropna(subset=["regime"]).reset_index(drop=True)

    return proxy


def classify_current_regime(**kwargs) -> dict:
    """
    Returns the current (most recent) regime classification plus context.
    """
    history = compute_regime_history(**kwargs)
    if history.empty:
        return {"regime": "UNKNOWN", "date": None, "rolling_return_pct": None,
                "rolling_vol_annualized_pct": None, "error": "no data"}

    latest = history.iloc[-1]
    return {
        "regime": latest["regime"],
        "date": str(pd.Timestamp(latest["date"]).date()),
        "rolling_return_pct": float(latest["rolling_return_pct"]),
        "rolling_vol_annualized_pct": float(latest["rolling_vol_annualized_pct"]),
    }


def regime_distribution(history: pd.DataFrame) -> dict:
    """Summary stats: how many days in each regime."""
    counts = history["regime"].value_counts().to_dict()
    total = len(history)
    return {
        "counts": counts,
        "percentages": {k: round(v / total * 100, 1) for k, v in counts.items()},
        "total_days": total,
    }


def save_regime_history(history: pd.DataFrame) -> Path:
    """Persist regime history to CSV."""
    history.to_csv(config.REGIME_REPORT_FILE, index=False)
    logger.info("Regime history written to %s (%d rows)", config.REGIME_REPORT_FILE, len(history))
    return config.REGIME_REPORT_FILE
