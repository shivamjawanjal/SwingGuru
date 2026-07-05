# NSE Swing Trading Pipeline

## Phase 1 — Data
Universe Selection -> Bhavcopy Download -> OHLCV Build -> Validation

## Phase 2 — Feature Engineering
Validated OHLCV -> Trend / Momentum / Volatility / Volume / Price
Structure / Candlestick features -> one enriched CSV per symbol

## Phase 3 — Label Generation
Triple-barrier labeling: for each entry, does price hit the profit
target before the stop loss within the holding window?

## Phase 4 — Dataset Builder
Joins features + labels per symbol, normalizes absolute-price columns,
concatenates every symbol into one master training table.

## Phase 5 — Machine Learning
Trains and compares 5 model families on a chronological train/val
split: Logistic Regression, Random Forest, XGBoost, LightGBM,
CatBoost.

## Phase 6 — Evaluation
Runs explicit leakage checks, then evaluates the Phase 5 winner on the
held-out test set for the first genuinely unbiased read of performance.

## Phase 7 — Backtesting
Event-driven simulation of actually trading the model's signals:
finite capital, position limits, brokerage + slippage, daily
mark-to-market equity curve.

## Phase 8 — Daily Scanner
Refreshes data, scores every symbol's latest row, outputs the Top-N
ranked swing trade candidates.

## Advanced — Walk-Forward Validation
Retrains the winning model across expanding-window folds spanning the
whole dataset, for a multi-regime honesty check beyond one static split.

## Phase 9 — Risk Management
Replaces Phase 7's equal-weight sizing and fixed stop with ATR-based
stops, a trailing stop, 1%-risk position sizing, and cap-bucket
exposure limits.

## Setup
    pip install -r requirements.txt

## Run Phase 1 (everything)
    python run_phase1.py

## Run a Phase 1 subset
    python run_phase1.py --steps universe
    python run_phase1.py --steps download --start 2020-01-01 --end 2024-01-01
    python run_phase1.py --steps build_ohlcv --workers 8
    python run_phase1.py --steps validate

## Run Phase 2
    python run_phase2.py
    python run_phase2.py --workers 8

Phase 2 reads data/validated_symbols.csv if present (Phase 1 Step 4
output) and otherwise falls back to every file in data/ohlcv/.
Output lands in data/features/<SYMBOL>.csv — 59 feature columns per
symbol (EMAs, RSI, MACD, ROC, ATR, Bollinger, historical volatility,
OBV, CMF, rolling VWAP, volume ratio, HH/HL structure, gaps,
breakouts, support/resistance, and Doji/Hammer/Inside Bar/Engulfing
candlestick flags). Symbols with fewer than
`config.MIN_ROWS_FOR_FEATURES` (210) rows are skipped — there isn't
enough history to warm up EMA200 and the 20-day rolling windows
without producing garbage.

## Run Phase 3
    python run_phase3.py
    python run_phase3.py --profit 0.08 --stop 0.04 --days 15
    python run_phase3.py --workers 8

Triple-barrier labeling per symbol. Entry is priced at the NEXT
trading day's open after the signal day (config.ENTRY_LAG_DAYS),
never the signal day's own close — using same-day close as entry
would leak information you can't actually act on live. For each
entry, scans forward up to `--days` trading days: if price hits the
profit target first -> label=1, if it hits the stop first -> label=0,
if neither is touched by the end of the window -> label=0 (timeout).
Rows near the end of a symbol's history that don't have enough future
data to complete the window are left as label=NaN (censored) rather
than guessed — drop these before training. Output:
data/labels/<SYMBOL>.csv with entry/exit price, exit_reason
(profit_target/stop_loss/timeout), holding_days, realized_return_pct,
and the final binary label. Kept separate from data/features/ on
purpose — Phase 4 (Dataset Builder) is what joins the two; this
separation lets you re-label with different profit/stop/window
parameters without re-running feature engineering.

## Run Phase 4
    python run_phase4.py
    python run_phase4.py --workers 8

Merges each symbol's features + labels (inner join on date, censored
rows already dropped in Phase 3 output). Several Phase 2 columns are
in absolute rupee terms (EMA levels, MACD, momentum, resistance/
support, VWAP, OBV) — pooling those raw across symbols of very
different price scales would be meaningless (a MACD of 12 means
something different for a ₹50 stock than a ₹2500 stock). This step
normalizes MACD/macd_signal/macd_hist/momentum_10 to %-of-close and
keeps only the normalized/bounded/ratio/boolean columns
(`config.DATASET_FEATURE_COLUMNS`, currently 43 columns) in the
actual training feature set; raw absolute columns stay in the merged
file for reference but are excluded from the model input. Also joins
in each symbol's cap bucket (`primary_index` from
validated_symbols.csv) as a categorical column.

Output: data/datasets/master_dataset.parquet (one row per
symbol/date, sorted by date then symbol) and
data/datasets/dataset_manifest.json describing exactly which columns
are features, which is categorical, which is the label, and which
are metadata-only — Phase 5 reads this manifest instead of
hardcoding column names.

## Run Phase 5
    python run_phase5.py
    python run_phase5.py --workers 8
    python run_phase5.py --metric precision_at_top_20

Splits the master dataset CHRONOLOGICALLY (70/15/15 by calendar date,
never randomly — a random split would let the model train on
tomorrow and test on yesterday, which is the single most common way
retail trading ML projects fool themselves). Test set is written but
untouched here; Phase 6 evaluates on it.

Trains all 5 model families on train, scores each on val, and saves
whichever wins on `config.MODEL_SELECTION_METRIC` (default
`roc_auc`) to models/best_model.joblib + models/best_model_meta.json.
Full comparison across all 5 models (AUC, average precision,
precision/recall/F1 at threshold 0.5, and precision@top-20/50/100) is
written to reports/model_comparison.json.

Note precision@top-K and AUC can disagree — a model can rank the
*overall* population well (high AUC) while being unreliable at the
*very top* of its own ranking, which is what the daily scanner
actually uses ("Top 20 Swing Trades"). If a comparison run shows this
split, prefer selecting by `--metric precision_at_top_20` over the
AUC default. With only ~2000 validation rows this metric is noisy
(one flipped prediction moves precision@20 by 5 points) — it
stabilizes once you're running on the full 6-year real dataset.

## Run Phase 6
    python run_phase6.py

Re-derives the same chronological split Phase 5 used, then runs three
checks before trusting any number:
  - **Temporal overlap** (hard assertion): every train date strictly
    before every val date, every val date strictly before every test
    date. Raises immediately if violated.
  - **Duplicate rows across splits** (hard assertion): no
    (symbol, date) row appears in more than one split.
  - **Regime shift** (warning only): flags if the base positive rate
    differs by more than `config.REGIME_SHIFT_WARNING_THRESHOLD_PCT`
    between train/val/test — not a bug, but worth knowing before
    trusting the test numbers as representative of "normal" markets.

Then evaluates the saved model on test (untouched until this point)
and writes reports/test_evaluation_report.json: overall metrics,
confusion matrix, and a breakdown by symbol and by cap_bucket (a
model that looks fine in aggregate can still be quietly unreliable on
small-caps specifically). Also flags if test AUC comes in suspiciously
*higher* than the recorded validation AUC — real models rarely get
more confident on truly unseen data, so a big jump upward is a leak
smell, not good news.

## Run Phase 7
    python run_phase7.py
    python run_phase7.py --threshold 0.6 --max-positions 15

Takes every test-set signal above `--threshold` predicted probability
(default `config.BACKTEST_SIGNAL_PROBABILITY_THRESHOLD`), then
simulates trading them day by day rather than just averaging trade
outcomes:
  - Capital is finite — `--max-positions` caps concurrent open
    positions (default 10); when more candidates appear on a day than
    there are free slots, only the highest-probability ones get taken.
  - Long-only, one position per symbol at a time.
  - Brokerage (`config.BACKTEST_BROKERAGE_PCT_PER_SIDE`) and slippage
    (`config.BACKTEST_SLIPPAGE_PCT_PER_SIDE`) are applied on both
    entry and exit — signal price is not fill price.
  - Position size is 1/max-positions of CURRENT mark-to-market equity
    at entry time, so returns compound realistically instead of
    reusing the static starting capital for every trade.
  - Equity is marked to market daily using actual close prices from
    data/ohlcv/ for open positions, which is what makes drawdown and
    Sharpe meaningful rather than approximate.

Runs entirely on the held-out TEST set from Phase 6 (same
leakage-checked predictions) — this is not an in-sample result.

Output: backtests/<run_id>/trade_log.csv, equity_curve.csv, and
backtest_report.json with win rate, profit factor, max drawdown,
Sharpe ratio, CAGR, expectancy per trade, and average holding days.
Note this is Phase 7's basic capital allocation (equal-weight,
probability-ranked) — real risk management (ATR-based stops, 1%
risk-per-trade sizing, sector exposure caps) is Phase 9.

## Run Phase 8
    python run_phase8.py
    python run_phase8.py --skip-data-refresh --top-n 30

By default, refreshes data first (downloads any new bhavcopy in the
last `--lookback-days`, rebuilds OHLCV, rebuilds features — thin
wrappers around Phase 1/2, no new logic), then takes each symbol's
MOST RECENT feature row, scores it with the saved model, and prints
the Top-N ranked candidates with entry/stop/target/holding guidance.
`--skip-data-refresh` scans against whatever's already in
data/features/ (useful offline or for testing).

Entry price shown is a REFERENCE only (the latest close) — actual
execution happens at the next session's open, which isn't known yet
at scan time; this is the standard convention for a pre/post-market
scanner. Stoploss/target/holding-days come directly from the same
triple-barrier parameters used in labeling
(`config.TRIPLE_BARRIER_STOP_PCT/PROFIT_PCT/MAX_DAYS`), so what the
scanner promises matches what the model was actually trained and
backtested against.

If the newest date across all feature files is more than
`config.SCANNER_STALENESS_WARNING_DAYS` old, a loud warning prints —
this usually means the data refresh step didn't run or failed
silently, and the scan is ranking old signals as if they were fresh.

Output: reports/scans/scan_<date>.csv.

## Run Walk-Forward Validation (Advanced Phase)
    python run_walk_forward.py
    python run_walk_forward.py --model lightgbm --folds 5

A single train/val/test split (Phase 5/6) only tells you how the
model did in ONE slice of market history. This retrains the SAME
model family Phase 5 selected (via `train_single_model` — the
identical code path, not a reimplementation) across expanding-window
folds:

    Fold 1: Train [------]        Test [--]
    Fold 2: Train [---------]     Test    [--]
    Fold 3: Train [------------]  Test        [--]

Every fold's test predictions came from a model that had never seen
that date or anything after it — this is what "no future leakage"
means in practice, checked across multiple market regimes instead of
just once. Reports pooled out-of-fold metrics (all folds' predictions
combined) and fold-to-fold AUC stability (mean/std/min/max) — a model
whose AUC swings wildly between folds is not one you should trust to
behave consistently live, even if its average looks fine. Output:
reports/walk_forward_report.json.

## Run Phase 9
    python run_phase9.py
    python run_phase9.py --threshold 0.6 --max-positions 15 --max-per-bucket 3

Same test-set signals as Phase 7, different capital management:

  - **Dynamic exit**: instead of Phase 3's static +8%/-4% triple
    barrier, each candidate's exit is simulated day-by-day against its
    own real OHLCV using an ATR-based stop
    (`entry - ATR_STOP_MULTIPLIER x ATR`) that trails upward as price
    rises (`highest_close_since_entry - TRAILING_STOP_ATR_MULTIPLIER x ATR`,
    only ever moving up). A trade that rallies to +15% and pulls back
    exits near the peak instead of riding a static target back down.
    Set `config.ENABLE_TRAILING_STOP = False` to fall back to a plain
    fixed ATR stop.
  - **Risk-based sizing**: position size is chosen so a stop-out costs
    exactly `config.RISK_PER_TRADE_PCT` (default 1%) of current
    equity, using each stock's OWN volatility instead of Phase 7's
    equal 1/N split. A tight-stop stock gets more shares than a
    wide-stop one, at the same dollar risk. Capped at
    `config.MAX_POSITION_NOTIONAL_PCT_OF_EQUITY` (default 25%) so an
    unusually tight stop can't imply an oversized position.
  - **Cap-bucket exposure limit**: `--max-per-bucket` caps how many
    concurrent positions can come from the same cap_bucket
    (NIFTY50/100/200/Midcap/Smallcap), so the portfolio can't
    accidentally fill up on correlated names.

Because risk-based sizing means tight-stop stocks consume more
cash/slots per trade, Phase 9 typically executes FEWER total trades
than Phase 7 on the same candidate list — this is expected, not a
bug; check `n_trades_executed` and the `skipped` breakdown in the
report to see why. Compare `backtests/<run>/backtest_report.json`
(Phase 7) against `backtests/risk_managed/<run>/backtest_report.json`
(Phase 9) to see the actual before/after impact of risk management on
Sharpe and max drawdown — the doc's claim that risk management "can
improve returns more than changing the ML model" is directly testable
this way.

## Notes
- Requires network access to archives.nseindia.com and nsearchives.nseindia.com.
- NSE occasionally renames index constituent files and adjusts bhavcopy
  schema; if a step starts failing outright, check configs/config.py first.
- Re-running `download` is safe/cheap: already-downloaded days are skipped.
- Feature formulas and thresholds (RSI period, Bollinger std, gap
  threshold, hammer/doji ratios, etc.) all live in configs/config.py —
  tune there, not inside the feature modules.
- Same-day tie-break (both barriers touched on the same daily bar,
  since bhavcopy has no intraday sequencing) is resolved by assuming
  whichever barrier is numerically closer to that day's open was hit
  first. This is a heuristic — genuinely intraday data would remove
  the ambiguity entirely, but isn't available from EOD bhavcopy.
