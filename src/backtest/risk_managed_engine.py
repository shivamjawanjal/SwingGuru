"""
Phase 9 — Risk-Managed Backtest Engine

Same event-driven structure as Phase 7's engine, but replaces two of
its simplifications:

  1. Exit is DYNAMIC (ATR stop + trailing stop, simulated day-by-day
     against real OHLCV) instead of Phase 3's static triple-barrier
     exit. Computed once per candidate up front — the exit path only
     depends on that symbol's own price history, not on portfolio
     state, so there's no need to interleave it with the event loop.
  2. Position sizing is RISK-based (1% of equity / stop distance) with
     a max-notional safety cap, instead of equal-weight 1/N — plus a
     per-cap-bucket exposure limit so the portfolio can't fill up on
     correlated names.

Sizing uses the INITIAL ATR stop (known at entry time), never the
final trailing-stop price (which is only known in hindsight) — sizing
a trade using information from its own future would be leakage into
the position-sizing decision itself.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config
from src.evaluation.evaluate_test import get_test_predictions
from src.risk.dynamic_exit import simulate_dynamic_exit
from src.risk.position_sizing import position_size_by_risk
from src.risk.exposure import can_open_new_position
from src.backtest.metrics import compute_backtest_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("risk_managed_engine")


def _load_ohlcv_lookup(symbols) -> dict:
    lookup = {}
    for symbol in symbols:
        path = config.OHLCV_DIR / f"{symbol}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        lookup[symbol] = df[["open", "high", "low", "close"]]
    return lookup


def _precompute_exits(candidates: pd.DataFrame, ohlcv_lookup: dict) -> pd.DataFrame:
    """
    For every candidate, simulate its dynamic exit and its initial ATR
    stop (used for sizing). Rows whose exit can't be determined
    (missing OHLCV, or genuinely censored by end-of-data) are dropped.
    """
    records = []
    for _, cand in candidates.iterrows():
        symbol = cand["symbol"]
        ohlcv = ohlcv_lookup.get(symbol)
        if ohlcv is None:
            continue

        atr_pct = cand["atr_pct"]
        initial_stop = cand["entry_price"] * (1 - config.ATR_STOP_MULTIPLIER * (atr_pct / 100))

        exit_info = simulate_dynamic_exit(
            ohlcv, entry_date=cand["entry_date"], entry_price=cand["entry_price"],
            atr_pct_at_signal=atr_pct,
        )
        if exit_info is None:
            continue

        rec = cand.to_dict()
        rec["initial_atr_stop"] = initial_stop
        rec.update({f"dyn_{k}": v for k, v in exit_info.items()})
        records.append(rec)

    return pd.DataFrame(records)


def run_risk_managed_backtest(
    probability_threshold: Optional[float] = None,
    max_concurrent_positions: Optional[int] = None,
    max_per_cap_bucket: Optional[int] = None,
) -> dict:
    threshold = probability_threshold if probability_threshold is not None else config.BACKTEST_SIGNAL_PROBABILITY_THRESHOLD
    max_positions = max_concurrent_positions if max_concurrent_positions is not None else config.BACKTEST_MAX_CONCURRENT_POSITIONS
    max_per_bucket = max_per_cap_bucket if max_per_cap_bucket is not None else config.MAX_POSITIONS_PER_CAP_BUCKET

    logger.info("Loading test-set predictions (leakage-checked)...")
    test_df, model_meta, split_info, leakage_report = get_test_predictions()
    logger.info("Risk-managed backtest: model=%s, test period %s", model_meta["model_name"], split_info["test_date_range"])

    candidates = test_df[test_df["proba"] >= threshold].copy()
    if candidates.empty:
        raise RuntimeError(f"No candidates clear probability_threshold={threshold}.")

    ohlcv_lookup = _load_ohlcv_lookup(candidates["symbol"].unique())
    logger.info("Simulating dynamic (ATR + trailing stop) exits for %d candidates...", len(candidates))
    candidates = _precompute_exits(candidates, ohlcv_lookup)
    if candidates.empty:
        raise RuntimeError("No candidates produced a resolvable dynamic exit — check OHLCV coverage.")

    candidates = candidates.sort_values(["entry_date", "proba"], ascending=[True, False])
    candidates_by_entry_date = {d: grp for d, grp in candidates.groupby("entry_date")}

    all_event_dates = sorted(set(candidates["entry_date"]) | set(candidates["dyn_exit_date"]))
    close_lookup = {sym: df["close"] for sym, df in ohlcv_lookup.items()}

    # Load market regime history to enable regime-adaptive risk sizing
    from src.regime.regime_detector import compute_regime_history
    try:
        regime_df = compute_regime_history()
        # Convert date to string for reliable matching
        regime_map = {str(pd.Timestamp(row["date"]).date()): row["regime"] for _, row in regime_df.iterrows()}
        logger.info("Loaded regime history for adaptive risk sizing.")
    except Exception as exc:
        logger.warning("Could not load regime history, falling back to BULL regime for all dates: %s", exc)
        regime_map = {}

    cash = config.BACKTEST_STARTING_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    skipped_no_slot, skipped_duplicate_symbol, skipped_bucket_cap, skipped_degenerate_stop, skipped_regime = 0, 0, 0, 0, 0

    brokerage = config.BACKTEST_BROKERAGE_PCT_PER_SIDE
    slippage = config.BACKTEST_SLIPPAGE_PCT_PER_SIDE

    for day in all_event_dates:
        for symbol in list(open_positions.keys()):
            pos = open_positions[symbol]
            if pos["exit_date"] != day:
                continue

            effective_exit_price = pos["planned_exit_price"] * (1 - slippage)
            exit_notional = pos["shares"] * effective_exit_price
            exit_brokerage = exit_notional * brokerage
            cash += exit_notional - exit_brokerage

            net_pnl = (exit_notional - exit_brokerage) - (pos["entry_notional"] + pos["entry_brokerage"])
            net_return_pct = net_pnl / (pos["entry_notional"] + pos["entry_brokerage"]) * 100

            trade_log.append({
                "symbol": symbol, "cap_bucket": pos["cap_bucket"],
                "entry_date": pos["entry_date"], "exit_date": pos["exit_date"],
                "entry_price": pos["entry_price"], "exit_price": pos["planned_exit_price"],
                "exit_reason": pos["exit_reason"], "holding_days": pos["holding_days"],
                "shares": pos["shares"], "net_pnl": net_pnl, "net_return_pct": net_return_pct,
                "probability": pos["probability"], "risk_amount": pos["risk_amount"],
                "capped_by_max_notional": pos["capped_by_max_notional"],
            })
            del open_positions[symbol]

        todays_candidates = candidates_by_entry_date.get(day, pd.DataFrame())
        if not todays_candidates.empty:
            mtm_open_value = sum(
                p["shares"] * close_lookup.get(sym, pd.Series(dtype=float)).get(day, p["entry_price"])
                for sym, p in open_positions.items()
            )
            current_equity = cash + mtm_open_value

            # Lookup current regime risk multiplier for the signal date
            day_str = str(pd.Timestamp(day).date())
            regime = regime_map.get(day_str, "UNKNOWN")
            regime_mult = config.REGIME_RISK_MULTIPLIERS.get(regime, 1.0)

            for _, cand in todays_candidates.iterrows():
                symbol = cand["symbol"]
                cap_bucket = cand.get("cap_bucket", "UNKNOWN")

                if symbol in open_positions:
                    skipped_duplicate_symbol += 1
                    continue
                if len(open_positions) >= max_positions:
                    skipped_no_slot += 1
                    continue
                if not can_open_new_position(open_positions, cap_bucket, max_per_bucket):
                    skipped_bucket_cap += 1
                    continue
                
                # Regime-adaptive deactivation filter
                if regime_mult == 0.0:
                    skipped_regime += 1
                    continue

                effective_entry_price = cand["entry_price"] * (1 + slippage)
                
                # Scale risk percentage dynamically by the regime multiplier
                sizing = position_size_by_risk(
                    current_equity, 
                    effective_entry_price, 
                    cand["initial_atr_stop"],
                    risk_pct=config.RISK_PER_TRADE_PCT * regime_mult
                )
                if sizing["shares"] <= 0:
                    skipped_degenerate_stop += 1
                    continue

                entry_notional = sizing["notional"]
                entry_brokerage = entry_notional * brokerage
                if entry_notional + entry_brokerage > cash:
                    skipped_no_slot += 1
                    continue

                cash -= (entry_notional + entry_brokerage)
                open_positions[symbol] = {
                    "entry_date": cand["entry_date"], "entry_price": cand["entry_price"],
                    "exit_date": cand["dyn_exit_date"], "planned_exit_price": cand["dyn_exit_price"],
                    "exit_reason": cand["dyn_exit_reason"], "holding_days": cand["dyn_holding_days"],
                    "shares": sizing["shares"], "entry_notional": entry_notional,
                    "entry_brokerage": entry_brokerage, "probability": cand["proba"],
                    "cap_bucket": cap_bucket, "risk_amount": sizing["risk_amount"],
                    "capped_by_max_notional": sizing["capped_by_max_notional"],
                }

        mtm_open_value = sum(
            p["shares"] * close_lookup.get(sym, pd.Series(dtype=float)).get(day, p["entry_price"])
            for sym, p in open_positions.items()
        )
        equity_curve.append({"date": day, "equity": cash + mtm_open_value})

    for symbol, pos in open_positions.items():
        effective_exit_price = pos["planned_exit_price"] * (1 - slippage)
        exit_notional = pos["shares"] * effective_exit_price
        exit_brokerage = exit_notional * brokerage
        net_pnl = (exit_notional - exit_brokerage) - (pos["entry_notional"] + pos["entry_brokerage"])
        net_return_pct = net_pnl / (pos["entry_notional"] + pos["entry_brokerage"]) * 100
        trade_log.append({
            "symbol": symbol, "cap_bucket": pos["cap_bucket"],
            "entry_date": pos["entry_date"], "exit_date": pos["exit_date"],
            "entry_price": pos["entry_price"], "exit_price": pos["planned_exit_price"],
            "exit_reason": pos["exit_reason"] + "_forced_close", "holding_days": pos["holding_days"],
            "shares": pos["shares"], "net_pnl": net_pnl, "net_return_pct": net_return_pct,
            "probability": pos["probability"], "risk_amount": pos["risk_amount"],
            "capped_by_max_notional": pos["capped_by_max_notional"],
        })

    trade_log_df = pd.DataFrame(trade_log)
    equity_curve_df = pd.DataFrame(equity_curve)

    logger.info(
        "Simulation complete: %d trades, skipped: %d no-slot/cash, %d duplicate-symbol, "
        "%d bucket-cap, %d degenerate-stop, %d regime-filter",
        len(trade_log_df), skipped_no_slot, skipped_duplicate_symbol, skipped_bucket_cap, skipped_degenerate_stop, skipped_regime,
    )

    metrics = compute_backtest_metrics(trade_log_df, equity_curve_df)
    exit_reason_counts = trade_log_df["exit_reason"].value_counts().to_dict() if not trade_log_df.empty else {}
    logger.info("Risk-managed backtest metrics: %s", metrics)
    logger.info("Exit reason breakdown: %s", exit_reason_counts)

    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.RISK_MANAGED_BACKTESTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trade_log_df.to_csv(run_dir / "trade_log.csv", index=False)
    equity_curve_df.to_csv(run_dir / "equity_curve.csv", index=False)

    report = {
        "run_id": run_id,
        "model_name": model_meta["model_name"],
        "probability_threshold": threshold,
        "max_concurrent_positions": max_positions,
        "max_positions_per_cap_bucket": max_per_bucket,
        "risk_per_trade_pct": config.RISK_PER_TRADE_PCT,
        "atr_stop_multiplier": config.ATR_STOP_MULTIPLIER,
        "trailing_stop_atr_multiplier": config.TRAILING_STOP_ATR_MULTIPLIER,
        "enable_trailing_stop": config.ENABLE_TRAILING_STOP,
        "starting_capital": config.BACKTEST_STARTING_CAPITAL,
        "split_info": split_info,
        "n_candidates_above_threshold": len(candidates),
        "n_trades_executed": len(trade_log_df),
        "skipped": {
            "no_slot_or_cash": skipped_no_slot, "duplicate_symbol": skipped_duplicate_symbol,
            "bucket_cap": skipped_bucket_cap, "degenerate_stop": skipped_degenerate_stop,
        },
        "exit_reason_breakdown": exit_reason_counts,
        "metrics": metrics,
    }
    with open(run_dir / "backtest_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Risk-managed backtest artifacts written to %s", run_dir)
    return report


if __name__ == "__main__":
    run_risk_managed_backtest()
