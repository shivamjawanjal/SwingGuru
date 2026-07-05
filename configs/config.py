"""
Central configuration for the NSE Swing Trading Pipeline.
Every path, URL, and tunable knob lives here so nothing is hardcoded
in the actual logic modules.
"""

from pathlib import Path
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SYMBOLS_RAW_DIR = RAW_DIR / "symbols"
BHAVCOPY_RAW_DIR = RAW_DIR / "bhavcopy"
OHLCV_DIR = DATA_DIR / "ohlcv"
FEATURES_DIR = DATA_DIR / "features"
LOG_DIR = ROOT_DIR / "logs"

SYMBOLS_MASTER_FILE = DATA_DIR / "symbols.csv"
VALIDATED_SYMBOLS_FILE = DATA_DIR / "validated_symbols.csv"

for d in (SYMBOLS_RAW_DIR, BHAVCOPY_RAW_DIR, OHLCV_DIR, FEATURES_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# NSE index constituent lists (Universe Selection)
# NOTE: NSE occasionally renames these files. If a download starts failing,
# check https://www.nseindia.com/market-data/live-equity-market for the
# current filename under "Index" > "Downloads".
# ---------------------------------------------------------------------------
NSE_ARCHIVES_BASE = "https://archives.nseindia.com/content/indices"

INDEX_UNIVERSE = {
    "NIFTY50": f"{NSE_ARCHIVES_BASE}/ind_nifty50list.csv",
    "NIFTY100": f"{NSE_ARCHIVES_BASE}/ind_nifty100list.csv",
    "NIFTY200": f"{NSE_ARCHIVES_BASE}/ind_nifty200list.csv",
    "NIFTYMIDCAP150": f"{NSE_ARCHIVES_BASE}/ind_niftymidcap150list.csv",
    "NIFTYSMALLCAP250": f"{NSE_ARCHIVES_BASE}/ind_niftysmallcap250list.csv",
}

# ---------------------------------------------------------------------------
# Bhavcopy (OHLCV) download settings
# ---------------------------------------------------------------------------
# Modern NSE full bhavcopy (UDiFF format), one zip per trading day.
BHAVCOPY_URL_TEMPLATE = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)

YEARS_OF_HISTORY = 6
END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=365 * YEARS_OF_HISTORY)

# NSE blocks aggressive/naive scraping. Keep concurrency modest and always
# go through the rate-limited session in src/utils/http_client.py.
MAX_CONCURRENT_DOWNLOADS = 6
REQUEST_TIMEOUT_SECONDS = 15
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
MIN_DELAY_BETWEEN_REQUESTS = 0.25  # per-worker throttle

# NSE requires browser-like headers or it rejects requests outright.
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}
NSE_HOME_URL = "https://www.nseindia.com/"

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------
MIN_TRADING_DAYS = 500       # drop symbols with less history than this (recent IPOs)
MAX_ALLOWED_ZERO_VOLUME_RATIO = 0.15  # drop if >15% of days have zero volume (suspended)

# ---------------------------------------------------------------------------
# Phase 2: Feature engineering
# ---------------------------------------------------------------------------
EMA_PERIODS = [20, 50, 100, 200]
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ROC_PERIODS = [5, 10, 20]
MOMENTUM_PERIODS = [10]

ATR_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
HIST_VOL_PERIOD = 20        # rolling annualized historical volatility window
TRADING_DAYS_PER_YEAR = 252

CMF_PERIOD = 20
VWAP_PERIOD = 20             # rolling VWAP window (we only have daily bars, not intraday)
VOLUME_RATIO_PERIOD = 20     # today's volume vs N-day average volume

PRICE_STRUCTURE_LOOKBACK = 20   # window for breakout / support-resistance / HH-HL detection
GAP_THRESHOLD_PCT = 0.02        # >=2% open-vs-prev-close move counts as a gap

DOJI_BODY_RATIO = 0.1           # body <= 10% of the day's range
HAMMER_LOWER_SHADOW_MULT = 2.0  # lower shadow >= 2x body
HAMMER_UPPER_SHADOW_MAX_RATIO = 0.15  # upper shadow <= 15% of range

# Minimum rows needed before indicators stop being NaN-heavy garbage
# (longest lookback used above is EMA200 / HIST_VOL warm-up).
MIN_ROWS_FOR_FEATURES = 210

# Feature build parallelism (per-symbol processing is embarrassingly
# parallel — one process per symbol file).
FEATURE_BUILD_WORKERS = None   # None -> cpu_count() - 1

# ---------------------------------------------------------------------------
# Phase 3: Label generation (triple-barrier)
# ---------------------------------------------------------------------------
LABELS_DIR = DATA_DIR / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Doc's worked example: "within next 15 days, did price reach +8% before -4%?"
TRIPLE_BARRIER_PROFIT_PCT = 0.08
TRIPLE_BARRIER_STOP_PCT = 0.04
TRIPLE_BARRIER_MAX_DAYS = 15

# You can't enter at the close of the signal day (you haven't seen it
# happen yet in real trading) — entry is priced at the open ENTRY_LAG_DAYS
# trading days later. 1 = next day's open, the realistic default.
ENTRY_LAG_DAYS = 1

# If both the profit and stop barrier are touched on the SAME day (bhavcopy
# only gives us daily OHLC, not intraday sequencing), we can't know which
# was hit first. Resolve it by assuming price moved continuously from the
# day's open, so whichever barrier is numerically closer to that day's
# open must have been reached first. This is a heuristic, not a certainty.
SAME_DAY_TIE_BREAK = "closer_to_open"

LABEL_BUILD_WORKERS = None   # None -> cpu_count() - 1

# ---------------------------------------------------------------------------
# Phase 4: Dataset Builder
# ---------------------------------------------------------------------------
DATASET_DIR = DATA_DIR / "datasets"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
MASTER_DATASET_FILE = DATASET_DIR / "master_dataset.parquet"
DATASET_MANIFEST_FILE = DATASET_DIR / "dataset_manifest.json"

DATASET_BUILD_WORKERS = None   # None -> cpu_count() - 1

# Columns carried through purely for identification / debugging / later
# phases (backtest needs entry_price etc.) — NOT fed to the model.
DATASET_META_COLUMNS = [
    "symbol", "date", "cap_bucket",
    "open", "high", "low", "close", "volume",
    "entry_date", "entry_price", "exit_date", "exit_price",
    "exit_reason", "holding_days", "realized_return_pct",
]

# Absolute-rupee-scale columns from Phase 2 that must NOT be pooled
# across symbols as-is (a MACD of 12 means something different for a
# ₹50 stock than a ₹2500 stock). Each gets a normalized %-of-close
# counterpart computed in build_dataset.py; the raw column is kept in
# the merged file for reference but excluded from DATASET_FEATURE_COLUMNS.
COLUMNS_NEEDING_NORMALIZATION = ["macd", "macd_signal", "macd_hist", "momentum_10"]

# The actual training feature set: normalized / bounded / ratio /
# boolean columns only. This list is also written into
# dataset_manifest.json so Phase 5 doesn't have to guess which columns
# are safe to feed the model.
DATASET_FEATURE_COLUMNS = [
    # trend
    "dist_from_ema_20_pct", "dist_from_ema_50_pct", "dist_from_ema_100_pct", "dist_from_ema_200_pct",
    "ema_20_above_50", "ema_50_above_100", "ema_100_above_200", "ema_bullish_stack",
    # momentum
    "rsi_14", "macd_pct", "macd_signal_pct", "macd_hist_pct",
    "macd_bullish_cross", "macd_bearish_cross",
    "roc_5", "roc_10", "roc_20", "momentum_10_pct",
    # volatility
    "atr_pct", "bollinger_width_pct", "bollinger_pct_b", "hist_volatility_annualized_pct",
    # volume
    "cmf_20", "dist_from_vwap_pct", "volume_ratio",
    # price structure
    "higher_high", "higher_low", "lower_high", "lower_low",
    "uptrend_structure", "downtrend_structure",
    "gap_pct", "gap_up", "gap_down",
    "dist_from_resistance_pct", "dist_from_support_pct", "breakout_up", "breakout_down",
    # candlestick
    "doji", "inside_bar", "hammer", "bullish_engulfing", "bearish_engulfing",
]

DATASET_CATEGORICAL_COLUMNS = ["cap_bucket"]
DATASET_LABEL_COLUMN = "label"

# ---------------------------------------------------------------------------
# Phase 5: Machine Learning
# ---------------------------------------------------------------------------
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COMPARISON_FILE = REPORTS_DIR / "model_comparison.json"
BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"
BEST_MODEL_META_FILE = MODELS_DIR / "best_model_meta.json"

# Chronological split — NEVER random. Fractions are applied to the sorted
# UNIQUE dates in the dataset (not raw row counts), so the split boundary
# is a clean calendar cutoff even though different symbols contribute
# different row counts per day.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15   # held out untouched until Phase 6

RANDOM_STATE = 42

# "Top 20 swing trades" (doc's daily scanner spec) -> precision among the
# top-K ranked-by-probability picks is the metric that actually matters
# for this use case, more than raw accuracy.
TOP_K_FOR_PRECISION = 20

MODEL_TRAINING_WORKERS = -1  # -1 -> use all cores where the library supports it

# Which metric decides the "winner" among the 5 candidate models.
# roc_auc is the conventional default, but for this product the
# TOP_K_FOR_PRECISION metric may matter more (see README) — override
# here if a comparison run shows AUC and top-K precision disagreeing.
MODEL_SELECTION_METRIC = "roc_auc"

# ---------------------------------------------------------------------------
# Phase 6: Evaluation
# ---------------------------------------------------------------------------
TEST_EVALUATION_REPORT_FILE = REPORTS_DIR / "test_evaluation_report.json"
# If train/val/test base rates differ by more than this, flag a possible
# regime shift between periods rather than silently trusting the numbers.
REGIME_SHIFT_WARNING_THRESHOLD_PCT = 10.0

# ---------------------------------------------------------------------------
# Phase 7: Backtesting
# ---------------------------------------------------------------------------
BACKTESTS_DIR = ROOT_DIR / "backtests"
BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_STARTING_CAPITAL = 1_000_000.0     # ₹10 lakh, arbitrary reference size
BACKTEST_MAX_CONCURRENT_POSITIONS = 10       # capital is finite — can't take every signal
BACKTEST_SIGNAL_PROBABILITY_THRESHOLD = 0.5  # only trade signals the model is this confident on

# Real-world frictions. NSE delivery trades: brokerage is often near-zero
# for retail but STT + exchange charges + stamp duty + slippage add up.
# These are deliberately conservative defaults, not a precise cost model —
# tune to your actual broker/cost structure.
BACKTEST_BROKERAGE_PCT_PER_SIDE = 0.001    # 0.10% per side (buy and sell each)
BACKTEST_SLIPPAGE_PCT_PER_SIDE = 0.0005    # 0.05% per side — real fill is worse than the signal price

RISK_FREE_RATE_ANNUAL = 0.065  # ~ Indian 10Y G-Sec ballpark, for Sharpe ratio

# ---------------------------------------------------------------------------
# Phase 8: Daily Scanner
# ---------------------------------------------------------------------------
SCANNER_TOP_N = 20  # "Top 20 Swing Trades" per the doc's spec
SCANS_DIR = REPORTS_DIR / "scans"
SCANS_DIR.mkdir(parents=True, exist_ok=True)

# If the newest date across all feature files is older than this many
# calendar days, the scan is very likely running on stale data (data
# refresh step didn't run or failed) — warn loudly rather than quietly
# ranking yesterday's-yesterday's signals as if they were fresh.
SCANNER_STALENESS_WARNING_DAYS = 5

# ---------------------------------------------------------------------------
# Advanced Phase: Walk-Forward Validation
# ---------------------------------------------------------------------------
WALK_FORWARD_REPORT_FILE = REPORTS_DIR / "walk_forward_report.json"

# Expanding-window retraining across the WHOLE dataset (train+val+test
# all get folded in — this is a separate, stricter evaluation
# methodology, not a replacement for the Phase 5 single split). The
# first fold's training window is at least WALK_FORWARD_MIN_TRAIN_FRACTION
# of the full date range; whatever remains is divided into
# WALK_FORWARD_N_FOLDS equal-length test windows, with the training
# window expanding to include every prior fold before each retrain.
WALK_FORWARD_N_FOLDS = 5
WALK_FORWARD_MIN_TRAIN_FRACTION = 0.5

# ---------------------------------------------------------------------------
# Phase 9: Risk Management
# ---------------------------------------------------------------------------
# Position sizing: risk a fixed % of current equity per trade, sized by
# distance to stop — NOT equal-weight like Phase 7. A tight-stop trade
# gets more capital than a wide-stop one for the same dollar risk.
RISK_PER_TRADE_PCT = 0.01          # risk 1% of equity per trade (doc's spec)
MAX_POSITION_NOTIONAL_PCT_OF_EQUITY = 0.25   # safety cap: no single position > 25% of equity,
                                               # regardless of how tight its stop is

# ATR-based stop-loss: distance from entry is a multiple of the stock's
# OWN volatility (ATR), not a fixed % — a volatile stock gets a wider
# stop than a quiet one, at the same "riskiness" in ATR terms.
ATR_STOP_MULTIPLIER = 2.0

# Trailing stop: ratchets up as price rises, tracking a multiple of ATR
# below the highest close seen since entry. Never moves down. Set
# ENABLE_TRAILING_STOP=False to fall back to a plain fixed ATR stop.
ENABLE_TRAILING_STOP = True
TRAILING_STOP_ATR_MULTIPLIER = 2.5

# Sector/cap exposure cap: prevents the scanner/backtest from filling
# every slot with correlated same-bucket stocks (e.g. all smallcaps).
MAX_POSITIONS_PER_CAP_BUCKET = 4

RISK_MANAGED_BACKTESTS_DIR = BACKTESTS_DIR / "risk_managed"
RISK_MANAGED_BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Market Regime Detection
# ---------------------------------------------------------------------------
# Proxy index: equal-weighted basket of large-cap Nifty50 stocks that are
# reliably present in data/ohlcv/ — used to compute market-wide return and
# volatility for regime classification without needing index-level data.
REGIME_PROXY_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "HCLTECH",
    "SUNPHARMA", "TATAMOTORS", "NTPC", "TITAN", "ADANIENT",
]

# Rolling windows for regime computation (in trading days).
REGIME_RETURN_WINDOW = 20      # ~1 month of trading
REGIME_VOLATILITY_WINDOW = 20  # ~1 month of trading

# Thresholds for classifying the regime from the proxy index.
# Bull: rolling return > threshold AND vol < vol threshold
# Bear: rolling return < -threshold AND vol < vol threshold
# High-volatility: vol >= vol threshold (regardless of return)
# Sideways: everything else
REGIME_RETURN_THRESHOLD_PCT = 2.0      # ±2% over the rolling window
REGIME_VOLATILITY_THRESHOLD_PCT = 25.0  # annualized vol >= 25% -> high-vol regime

REGIME_REPORT_FILE = REPORTS_DIR / "regime_history.csv"

# Regime-adaptive risk sizing multipliers:
# 1.0 (Full size) in BULL markets
# 0.5 (Half size) in SIDEWAYS markets
# 0.25 (Quarter size) in HIGH_VOL markets
# 0.0 (Deactivated / No new trades) in BEAR markets
REGIME_RISK_MULTIPLIERS = {
    "BULL": 1.0,
    "SIDEWAYS": 0.5,
    "HIGH_VOL": 0.25,
    "BEAR": 0.0,
    "UNKNOWN": 1.0,  # Fallback
}

# ---------------------------------------------------------------------------
# SHAP Explainability
# ---------------------------------------------------------------------------
# Max background samples for the SHAP explainer (more = slower but more
# accurate baseline). TreeExplainer doesn't need this; LinearExplainer does.
SHAP_MAX_BACKGROUND_SAMPLES = 200

# Number of top feature drivers to show per prediction in scanner output.
SHAP_TOP_DRIVERS = 3

SHAP_REPORT_DIR = REPORTS_DIR / "shap"
SHAP_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Telegram Alerts Config
# ---------------------------------------------------------------------------
# Paste your token and chat ID here once generated.
# If these are blank, the alert system will skip sending.
import os
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8822345165:AAEYqJWXRg2m1pd1uA3XntJQXja50CVbw10")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5188672620")


# ---------------------------------------------------------------------------
# Paper Trading System
# ---------------------------------------------------------------------------
PAPER_TRADING_DIR = DATA_DIR / "paper_trading"
PAPER_TRADING_DIR.mkdir(parents=True, exist_ok=True)

PAPER_PORTFOLIO_FILE = PAPER_TRADING_DIR / "state.json"
PAPER_EQUITY_FILE = PAPER_TRADING_DIR / "equity_history.csv"
PAPER_STARTING_CASH = 1000000.0  # Default starting cash: 10 Lakhs (₹1,000,000)

