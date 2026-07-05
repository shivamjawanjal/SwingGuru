"""
Phase 1 / Step 3 — Turn raw daily bhavcopy zips into per-symbol OHLCV files.

data/raw/bhavcopy/2024/20240103.csv.zip  ┐
data/raw/bhavcopy/2024/20240104.csv.zip  ├─►  data/ohlcv/RELIANCE.csv
data/raw/bhavcopy/2024/20240105.csv.zip  ┘    data/ohlcv/TCS.csv
                                               data/ohlcv/INFY.csv
                                               ...

Parsing ~1500 daily zip files is CPU-bound (unzip + CSV parse + filter),
so this uses a multiprocessing.Pool instead of threads. Each worker
parses one day independently and returns a small filtered DataFrame;
the main process concatenates everything ONCE and writes each symbol's
file a single time — this avoids N processes fighting over the same
output file.

NSE has changed its bhavcopy column schema more than once over the
years, so column detection is defensive rather than hardcoded to one
format.
"""

import logging
import sys
import zipfile
import io
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional, Set

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_ohlcv")

# Maps our canonical column name -> possible NSE source column names
# (old bhavcopy format vs the newer UDiFF format).
COLUMN_ALIASES = {
    "symbol": ["TckrSymb", "SYMBOL"],
    "series": ["SctySrs", "SERIES"],
    "date": ["TradDt", "TIMESTAMP"],
    "open": ["OpnPric", "OPEN"],
    "high": ["HghPric", "HIGH"],
    "low": ["LwPric", "LOW"],
    "close": ["ClsPric", "CLOSE"],
    "volume": ["TtlTradgVol", "TOTTRDQTY"],
}


def _resolve_columns(df_columns) -> dict:
    """Find which alias is present for each canonical field."""
    resolved = {}
    cols = set(df_columns)
    for canon, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in cols:
                resolved[canon] = alias
                break
    return resolved


def _parse_one_zip(args) -> Optional[pd.DataFrame]:
    zip_path_str, universe_symbols = args
    zip_path = Path(zip_path_str)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                return None
            with zf.open(csv_names[0]) as f:
                raw = f.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", zip_path.name, exc)
        return None

    colmap = _resolve_columns(df.columns)
    required = {"symbol", "series", "date", "open", "high", "low", "close", "volume"}
    if not required.issubset(colmap.keys()):
        logger.warning(
            "%s: unrecognized bhavcopy schema (missing %s), skipping",
            zip_path.name, required - colmap.keys(),
        )
        return None

    df = df.rename(columns={v: k for k, v in colmap.items()})
    df = df[list(required)]

    # Keep only equity series and symbols in our universe.
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["series"] = df["series"].astype(str).str.strip()
    df = df[(df["series"] == "EQ") & (df["symbol"].isin(universe_symbols))]

    if df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False).dt.date
    df = df.dropna(subset=["date"])
    df = df.drop(columns=["series"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _load_universe_symbols() -> Set[str]:
    if not config.SYMBOLS_MASTER_FILE.exists():
        raise RuntimeError(
            f"{config.SYMBOLS_MASTER_FILE} not found. Run "
            "src/universe/fetch_universe.py first."
        )
    symbols_df = pd.read_csv(config.SYMBOLS_MASTER_FILE)
    return set(symbols_df["symbol"].astype(str).str.strip())


def build_ohlcv_files(n_workers: Optional[int] = None) -> dict:
    universe_symbols = _load_universe_symbols()
    zip_files = sorted(config.BHAVCOPY_RAW_DIR.rglob("*.csv.zip"))

    if not zip_files:
        raise RuntimeError(
            f"No bhavcopy files found under {config.BHAVCOPY_RAW_DIR}. "
            "Run src/data/bhavcopy_downloader.py first."
        )

    n_workers = n_workers or max(1, cpu_count() - 1)
    logger.info(
        "Parsing %d bhavcopy files with %d worker processes (universe: %d symbols)...",
        len(zip_files), n_workers, len(universe_symbols),
    )

    tasks = [(str(zp), universe_symbols) for zp in zip_files]
    frames = []
    with Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_parse_one_zip, tasks, chunksize=8), 1):
            if result is not None:
                frames.append(result)
            if i % 200 == 0 or i == len(tasks):
                logger.info("Parsed %d/%d bhavcopy files...", i, len(tasks))

    if not frames:
        raise RuntimeError(
            "No usable rows extracted from any bhavcopy file — check that "
            "the schema in COLUMN_ALIASES still matches NSE's current format."
        )

    logger.info("Concatenating %d daily frames...", len(frames))
    all_data = pd.concat(frames, ignore_index=True)
    all_data = all_data.drop_duplicates(subset=["symbol", "date"])
    all_data = all_data.sort_values(["symbol", "date"])

    written = 0
    for symbol, group in all_data.groupby("symbol"):
        out_path = config.OHLCV_DIR / f"{symbol}.csv"
        group_out = group.drop(columns=["symbol"]).sort_values("date")
        group_out.to_csv(out_path, index=False)
        written += 1

    logger.info("Wrote OHLCV files for %d symbols to %s", written, config.OHLCV_DIR)
    return {
        "zip_files_parsed": len(zip_files),
        "usable_daily_frames": len(frames),
        "symbols_written": written,
        "total_rows": len(all_data),
    }


if __name__ == "__main__":
    build_ohlcv_files()
