"""
Phase 8 — Daily Scanner

Every evening (per the doc): take the latest data, generate features,
run the model, output the Top-N ranked swing trade candidates.

This reuses Phase 1/2's data-refresh functions, Phase 4's normalization
logic (so live features match training features exactly), and the
Phase 5 saved model — nothing here re-derives logic that already
exists elsewhere.
"""

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config
from src.dataset.merge_symbol import _normalize_absolute_columns
from src.dataset.build_dataset import _load_cap_buckets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("daily_scanner")


def refresh_data(lookback_days: int = 10) -> None:
    """
    Downloads any bhavcopy days from the last `lookback_days` that
    aren't cached yet, rebuilds OHLCV, and rebuilds features. Needs
    network access to nseindia.com — this is a thin wrapper around
    Phase 1/2, not new logic.
    """
    from src.data.bhavcopy_downloader import download_bhavcopy_range
    from src.data.build_ohlcv import build_ohlcv_files
    from src.features.build_features import build_features

    end = date.today()
    start = end - timedelta(days=lookback_days)
    logger.info("Refreshing bhavcopy for %s to %s...", start, end)
    download_bhavcopy_range(start=start, end=end)
    logger.info("Rebuilding OHLCV files...")
    build_ohlcv_files()
    logger.info("Rebuilding feature files...")
    build_features()


def _get_latest_feature_row_per_symbol() -> pd.DataFrame:
    """One row per symbol: its most recent date's feature values."""
    rows = []
    for path in sorted(config.FEATURES_DIR.glob("*.csv")):
        symbol = path.stem
        df = pd.read_csv(path, parse_dates=["date"])
        if df.empty:
            continue
        latest = df.sort_values("date").iloc[[-1]].copy()
        latest.insert(0, "symbol", symbol)
        rows.append(latest)

    if not rows:
        raise RuntimeError(f"No feature files found in {config.FEATURES_DIR}. Run Phase 2 first.")

    return pd.concat(rows, ignore_index=True)


def _check_staleness(latest_df: pd.DataFrame) -> dict:
    most_recent_date = latest_df["date"].max()
    days_old = (pd.Timestamp.today().normalize() - most_recent_date).days
    is_stale = days_old > config.SCANNER_STALENESS_WARNING_DAYS
    if is_stale:
        logger.warning(
            "Most recent data is %d days old (%s) — this exceeds the %d-day "
            "staleness threshold. Did the data refresh step run/succeed?",
            days_old, most_recent_date.date(), config.SCANNER_STALENESS_WARNING_DAYS,
        )
    return {"most_recent_date": str(most_recent_date.date()), "days_old": days_old, "is_stale": is_stale}


def run_daily_scan(
    top_n: Optional[int] = None,
    skip_data_refresh: bool = False,
    lookback_days: int = 10,
    explain: bool = True,
) -> pd.DataFrame:
    top_n = top_n if top_n is not None else config.SCANNER_TOP_N

    if not skip_data_refresh:
        refresh_data(lookback_days=lookback_days)
    else:
        logger.info("Skipping data refresh (--skip-data-refresh) — scanning against existing files.")

    if not config.BEST_MODEL_FILE.exists():
        raise RuntimeError(f"{config.BEST_MODEL_FILE} not found. Run Phase 5 first.")
    with open(config.BEST_MODEL_META_FILE) as f:
        model_meta = json.load(f)
    model = joblib.load(config.BEST_MODEL_FILE)

    logger.info("Loading latest feature row per symbol...")
    latest_df = _get_latest_feature_row_per_symbol()
    staleness = _check_staleness(latest_df)

    latest_df = _normalize_absolute_columns(latest_df)
    cap_buckets = _load_cap_buckets()
    latest_df["cap_bucket"] = latest_df["symbol"].map(cap_buckets).fillna("UNKNOWN")

    feature_columns = model_meta["feature_columns"]
    categorical_columns = model_meta["categorical_columns"]
    missing = set(feature_columns) - set(latest_df.columns)
    if missing:
        raise RuntimeError(
            f"Latest feature rows are missing expected columns: {missing}. "
            "Feature engineering (Phase 2) may be out of sync with the trained model."
        )

    # Drop symbols with NaN in any feature column
    nan_mask = latest_df[feature_columns].isna().any(axis=1)
    if nan_mask.any():
        nan_symbols = latest_df.loc[nan_mask, "symbol"].tolist()
        logger.warning(
            "Dropping %d symbols from scan because their latest row contains NaNs in feature columns: %s",
            len(nan_symbols), nan_symbols[:10]
        )
        latest_df = latest_df[~nan_mask].reset_index(drop=True)

    X = latest_df[feature_columns + categorical_columns].copy()
    if model_meta["uses_native_categorical"]:
        for col in categorical_columns:
            X[col] = X[col].astype(str).astype("category")
    else:
        X = pd.get_dummies(X, columns=categorical_columns)

    X = X.reindex(columns=model_meta["feature_names_after_encoding"], fill_value=0)
    latest_df["probability"] = model.predict_proba(X)[:, 1]

    ranked = latest_df.sort_values("probability", ascending=False).head(top_n).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    ranked["entry_ref_price"] = ranked["close"]
    ranked["stop_loss"] = ranked["entry_ref_price"] * (1 - config.TRIPLE_BARRIER_STOP_PCT)
    ranked["target"] = ranked["entry_ref_price"] * (1 + config.TRIPLE_BARRIER_PROFIT_PCT)
    ranked["max_holding_days"] = config.TRIPLE_BARRIER_MAX_DAYS

    # --- Market regime ---
    regime_info = {"regime": "UNKNOWN"}
    try:
        from src.regime.regime_detector import classify_current_regime
        regime_info = classify_current_regime()
        ranked["regime"] = regime_info["regime"]
        logger.info("Current market regime: %s", regime_info["regime"])
    except Exception as exc:
        logger.warning("Regime detection failed (non-fatal): %s", exc)
        ranked["regime"] = "UNKNOWN"

    # --- SHAP explainability ---
    if explain:
        try:
            from src.explainability.shap_explainer import explain_predictions
            # Get the X rows corresponding to the top-N ranked symbols.
            ranked_indices = ranked.index
            X_ranked = X.iloc[
                latest_df.sort_values("probability", ascending=False).head(top_n).index
            ].reset_index(drop=True)
            # Load background data if available (needed for LinearExplainer)
            bg_path = config.MODELS_DIR / "bg_sample.joblib"
            background = joblib.load(bg_path) if bg_path.exists() else None

            shap_result = explain_predictions(
                model=model,
                model_name=model_meta["model_name"],
                X=X_ranked,
                feature_names=model_meta["feature_names_after_encoding"],
                background_data=background,
            )
            ranked["top_drivers"] = shap_result["top_drivers"]
        except Exception as exc:
            logger.warning("SHAP explainability failed (non-fatal): %s", exc)
            ranked["top_drivers"] = ""
    else:
        ranked["top_drivers"] = ""

    report_cols = [
        "rank", "symbol", "cap_bucket", "probability",
        "entry_ref_price", "stop_loss", "target", "max_holding_days",
        "regime", "top_drivers", "date",
    ]
    report = ranked[report_cols].rename(columns={"date": "signal_date"})

    scan_date_str = staleness["most_recent_date"]
    out_path = config.SCANS_DIR / f"scan_{scan_date_str}.csv"
    report.to_csv(out_path, index=False)

    logger.info("Scan complete. Data as of %s (%d days old). Report saved to %s",
                staleness["most_recent_date"], staleness["days_old"], out_path)

    _print_cli_table(report, staleness, model_meta["model_name"], regime_info)

    # Send daily swing candidates to Telegram
    try:
        from src.utils.telegram_alerts import send_telegram_message
        regime_str = regime_info.get("regime", "UNKNOWN")
        regime_emoji = {"BULL": "🟢", "BEAR": "🔴", "HIGH_VOL": "🟡", "SIDEWAYS": "⚪"}.get(regime_str, "❓")
        
        msg = f"<b>🎯 TOP SWING CANDIDATES ({scan_date_str})</b>\n"
        msg += f"Model: <code>{model_meta['model_name']}</code> | Regime: {regime_emoji} <b>{regime_str}</b>\n\n"
        
        for _, r in report.head(10).iterrows():
            drivers = f" ({r['top_drivers']})" if r.get("top_drivers") else ""
            msg += (
                f"<b>{r['rank']}. {r['symbol']}</b> | Prob: <b>{r['probability']*100:.1f}%</b>\n"
                f"Entry: <code>{r['entry_ref_price']:.2f}</code> | Target: <code>{r['target']:.2f}</code> | Stop: <code>{r['stop_loss']:.2f}</code>\n"
                f"Drivers: <i>{drivers}</i>\n\n"
            )
        send_telegram_message(msg)
    except Exception as exc:
        logger.warning("Failed to send Telegram daily scan notification: %s", exc)

    return report


def _print_cli_table(report: pd.DataFrame, staleness: dict, model_name: str, regime_info: dict = None) -> None:
    regime_str = regime_info.get("regime", "UNKNOWN") if regime_info else "UNKNOWN"
    regime_emoji = {"BULL": "[+]", "BEAR": "[-]", "HIGH_VOL": "[!]", "SIDEWAYS": "[~]"}.get(regime_str, "[?]")
    print()
    print("=" * 120)
    print(f"  TOP {len(report)} SWING TRADE CANDIDATES  |  model={model_name}  |  "
          f"data as of {staleness['most_recent_date']} ({staleness['days_old']}d old)  |  "
          f"regime={regime_emoji} {regime_str}")
    if staleness["is_stale"]:
        print(f"  *** WARNING: data is stale (> {config.SCANNER_STALENESS_WARNING_DAYS} days old) ***")
    print("=" * 120)
    has_drivers = "top_drivers" in report.columns and report["top_drivers"].str.len().sum() > 0
    if has_drivers:
        header = f"{'Rk':>3} {'Symbol':<12} {'Cap':<18} {'Prob':>6} {'Entry':>10} {'Stop':>10} {'Target':>10} {'Days':>5}  {'Why (top drivers)'}"
    else:
        header = f"{'Rk':>3} {'Symbol':<12} {'Cap':<18} {'Prob':>6} {'Entry':>10} {'Stop':>10} {'Target':>10} {'Days':>5}"
    print(header)
    print("-" * 120)
    for _, r in report.iterrows():
        base = (
            f"{r['rank']:>3} {r['symbol']:<12} {r['cap_bucket']:<18} "
            f"{r['probability']*100:>5.1f}% {r['entry_ref_price']:>10.2f} "
            f"{r['stop_loss']:>10.2f} {r['target']:>10.2f} {r['max_holding_days']:>5}"
        )
        if has_drivers:
            drivers = r.get('top_drivers', '')
            base += f"  {drivers}"
        print(base)
    print("=" * 120)
    print()


if __name__ == "__main__":
    run_daily_scan()
