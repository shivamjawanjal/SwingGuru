"""
Phase 7 — Backtest Engine

Simulates actually trading the model's signals day by day, not just
averaging trade outcomes in isolation. This matters because:

  - Capital is finite: you can't take every signal, so when there are
    more candidates than open slots on a given day, only the top-N by
    predicted probability get taken.
  - Positions overlap in time across symbols, so an accurate equity
    curve (needed for drawdown and Sharpe) requires daily
    mark-to-market of open positions, not just a list of independent
    trade P&Ls.
  - Real fills are worse than the signal price: brokerage + slippage
    are applied on both entry and exit.

Trade outcomes (entry/exit price, exit reason, holding days) already
come from the Phase 3 triple-barrier labels — this engine's job is
capital allocation and cost modeling around those outcomes, not
re-deriving them.
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
from src.backtest.metrics import compute_backtest_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backtest_engine")


def _load_close_price_lookup(symbols) -> dict:
    """symbol -> pandas Series indexed by date of close price, for mark-to-market."""
    lookup = {}
    for symbol in symbols:
        path = config.OHLCV_DIR / f"{symbol}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        lookup[symbol] = df.set_index("date")["close"]
    return lookup


def run_backtest(
    probability_threshold: Optional[float] = None,
    max_concurrent_positions: Optional[int] = None,
) -> dict:
    threshold = probability_threshold if probability_threshold is not None else config.BACKTEST_SIGNAL_PROBABILITY_THRESHOLD
    max_positions = max_concurrent_positions if max_concurrent_positions is not None else config.BACKTEST_MAX_CONCURRENT_POSITIONS

    logger.info("Loading test-set predictions (leakage-checked)...")
    test_df, model_meta, split_info, leakage_report = get_test_predictions()
    logger.info("Backtesting model=%s over test period %s", model_meta["model_name"], split_info["test_date_range"])

    candidates = test_df[test_df["proba"] >= threshold].copy()
    candidates = candidates.sort_values(["entry_date", "proba"], ascending=[True, False])
    logger.info(
        "%d / %d test rows clear the probability threshold (%.2f) and are tradeable candidates",
        len(candidates), len(test_df), threshold,
    )
    if candidates.empty:
        raise RuntimeError(
            f"No candidates clear probability_threshold={threshold}. "
            "Try a lower threshold or check the model isn't degenerate."
        )

    close_lookup = _load_close_price_lookup(candidates["symbol"].unique())

    candidates_by_entry_date = {
        d: grp for d, grp in candidates.groupby("entry_date")
    }

    all_event_dates = sorted(
        set(candidates["entry_date"]) | set(candidates["exit_date"])
    )

    cash = config.BACKTEST_STARTING_CAPITAL
    open_positions = {}  # symbol -> position dict (long-only, one position per symbol at a time)
    trade_log = []
    equity_curve = []
    skipped_no_slot, skipped_duplicate_symbol = 0, 0

    brokerage = config.BACKTEST_BROKERAGE_PCT_PER_SIDE
    slippage = config.BACKTEST_SLIPPAGE_PCT_PER_SIDE

    for day in all_event_dates:
        # --- 1. Close positions scheduled to exit today ---
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
                "symbol": symbol,
                "entry_date": pos["entry_date"], "exit_date": pos["exit_date"],
                "entry_price": pos["entry_price"], "exit_price": pos["planned_exit_price"],
                "exit_reason": pos["exit_reason"], "holding_days": pos["holding_days"],
                "shares": pos["shares"], "net_pnl": net_pnl, "net_return_pct": net_return_pct,
                "probability": pos["probability"],
            })
            del open_positions[symbol]

        # --- 2. Open new positions from today's candidates ---
        todays_candidates = candidates_by_entry_date.get(day, pd.DataFrame())
        if not todays_candidates.empty:
            # Mark-to-market current equity BEFORE sizing new positions,
            # so position size reflects compounding, not the static starting capital.
            mtm_open_value = sum(
                p["shares"] * close_lookup.get(sym, pd.Series(dtype=float)).get(day, p["entry_price"])
                for sym, p in open_positions.items()
            )
            current_equity = cash + mtm_open_value

            for _, cand in todays_candidates.iterrows():
                symbol = cand["symbol"]
                if symbol in open_positions:
                    skipped_duplicate_symbol += 1
                    continue  # long-only, one position per symbol at a time
                if len(open_positions) >= max_positions:
                    skipped_no_slot += 1
                    continue

                position_notional = current_equity / max_positions
                effective_entry_price = cand["entry_price"] * (1 + slippage)
                if effective_entry_price <= 0:
                    continue
                shares = position_notional / effective_entry_price
                entry_brokerage = position_notional * brokerage

                if position_notional + entry_brokerage > cash:
                    skipped_no_slot += 1
                    continue  # not enough free cash despite a nominal "slot" being open

                cash -= (position_notional + entry_brokerage)
                open_positions[symbol] = {
                    "entry_date": cand["entry_date"], "entry_price": cand["entry_price"],
                    "exit_date": cand["exit_date"], "planned_exit_price": cand["exit_price"],
                    "exit_reason": cand["exit_reason"], "holding_days": cand["holding_days"],
                    "shares": shares, "entry_notional": position_notional,
                    "entry_brokerage": entry_brokerage, "probability": cand["proba"],
                }

        # --- 3. Mark-to-market equity snapshot for today ---
        mtm_open_value = sum(
            p["shares"] * close_lookup.get(sym, pd.Series(dtype=float)).get(day, p["entry_price"])
            for sym, p in open_positions.items()
        )
        equity_curve.append({"date": day, "equity": cash + mtm_open_value})

    # Force-close anything still open at the end of the simulation window
    # (shouldn't normally happen since every candidate has a known exit_date
    # within the event calendar, but guards against edge-of-data truncation).
    for symbol, pos in open_positions.items():
        effective_exit_price = pos["planned_exit_price"] * (1 - slippage)
        exit_notional = pos["shares"] * effective_exit_price
        exit_brokerage = exit_notional * brokerage
        net_pnl = (exit_notional - exit_brokerage) - (pos["entry_notional"] + pos["entry_brokerage"])
        net_return_pct = net_pnl / (pos["entry_notional"] + pos["entry_brokerage"]) * 100
        trade_log.append({
            "symbol": symbol, "entry_date": pos["entry_date"], "exit_date": pos["exit_date"],
            "entry_price": pos["entry_price"], "exit_price": pos["planned_exit_price"],
            "exit_reason": pos["exit_reason"] + "_forced_close", "holding_days": pos["holding_days"],
            "shares": pos["shares"], "net_pnl": net_pnl, "net_return_pct": net_return_pct,
            "probability": pos["probability"],
        })

    trade_log_df = pd.DataFrame(trade_log)
    equity_curve_df = pd.DataFrame(equity_curve)

    logger.info(
        "Simulation complete: %d trades executed, %d skipped (no slot/cash), "
        "%d skipped (duplicate symbol already held)",
        len(trade_log_df), skipped_no_slot, skipped_duplicate_symbol,
    )

    metrics = compute_backtest_metrics(trade_log_df, equity_curve_df)
    logger.info("Backtest metrics: %s", metrics)

    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.BACKTESTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trade_log_df.to_csv(run_dir / "trade_log.csv", index=False)
    equity_curve_df.to_csv(run_dir / "equity_curve.csv", index=False)

    report = {
        "run_id": run_id,
        "model_name": model_meta["model_name"],
        "probability_threshold": threshold,
        "max_concurrent_positions": max_positions,
        "starting_capital": config.BACKTEST_STARTING_CAPITAL,
        "brokerage_pct_per_side": brokerage,
        "slippage_pct_per_side": slippage,
        "split_info": split_info,
        "n_candidates_above_threshold": len(candidates),
        "n_trades_executed": len(trade_log_df),
        "n_skipped_no_slot_or_cash": skipped_no_slot,
        "n_skipped_duplicate_symbol": skipped_duplicate_symbol,
        "metrics": metrics,
    }
    with open(run_dir / "backtest_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Backtest artifacts written to %s", run_dir)
    return report


if __name__ == "__main__":
    run_backtest()
