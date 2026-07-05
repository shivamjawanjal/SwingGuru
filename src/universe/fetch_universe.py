"""
Phase 1 / Step 1 — Universe Selection

Downloads the constituent lists for Nifty 50 / 100 / 200 / Midcap150 /
Smallcap250, tags every symbol with which index(es) it belongs to, and
writes a single master data/symbols.csv.

Downloads run concurrently (they're independent, tiny files, and this
is purely I/O-bound) via a ThreadPoolExecutor.
"""

import csv
import io
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config
from src.utils import http_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_universe")


def _download_index_list(index_name: str, url: str) -> List[Dict[str, str]]:
    """Download one index constituent CSV and return parsed rows."""
    logger.info("Downloading %s constituent list...", index_name)
    resp = http_client.get(url)
    if resp is None:
        logger.error("Failed to download %s from %s", index_name, url)
        return []

    # Cache the raw file for auditability / offline reprocessing.
    raw_path = config.SYMBOLS_RAW_DIR / f"{index_name}.csv"
    raw_path.write_bytes(resp.content)

    text = resp.content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    rows = []
    for row in reader:
        # NSE's index CSVs use "Symbol" and "Series" as column names.
        symbol = (row.get("Symbol") or "").strip()
        series = (row.get("Series") or "EQ").strip()
        company = (row.get("Company Name") or "").strip()
        if not symbol:
            continue
        rows.append({"symbol": symbol, "series": series, "company_name": company})

    logger.info("%s: %d symbols parsed", index_name, len(rows))
    return rows


def build_symbol_universe() -> Path:
    """
    Downloads all configured index lists concurrently, merges them,
    and writes the deduplicated master symbols.csv.

    Cap classification priority (most exclusive wins, in this order):
    NIFTY50 > NIFTY100 > NIFTY200 > NIFTYMIDCAP150 > NIFTYSMALLCAP250
    A symbol keeps a record of every index it appears in, plus a
    single 'primary_index' for quick cap-bucket filtering later.
    """
    priority_order = [
        "NIFTY50", "NIFTY100", "NIFTY200",
        "NIFTYMIDCAP150", "NIFTYSMALLCAP250",
    ]

    results: Dict[str, List[Dict[str, str]]] = {}

    with ThreadPoolExecutor(max_workers=min(len(config.INDEX_UNIVERSE), 5)) as pool:
        futures = {
            pool.submit(_download_index_list, name, url): name
            for name, url in config.INDEX_UNIVERSE.items()
        }
        for future in as_completed(futures):
            index_name = futures[future]
            try:
                results[index_name] = future.result()
            except Exception as exc:
                logger.error("Unhandled error downloading %s: %s", index_name, exc)
                results[index_name] = []

    # Merge: symbol -> {company_name, series, indices: set()}
    merged: Dict[str, Dict] = {}
    for index_name in priority_order:
        for row in results.get(index_name, []):
            sym = row["symbol"]
            if sym not in merged:
                merged[sym] = {
                    "company_name": row["company_name"],
                    "series": row["series"],
                    "indices": set(),
                }
            merged[sym]["indices"].add(index_name)

    if not merged:
        raise RuntimeError(
            "No symbols were downloaded from any index. Check network "
            "access to archives.nseindia.com and verify the filenames "
            "in configs/config.py:INDEX_UNIVERSE are still current."
        )

    # Assign primary_index by priority order (most exclusive / highest cap wins).
    with open(config.SYMBOLS_MASTER_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "company_name", "series", "primary_index", "all_indices"])
        for sym in sorted(merged):
            info = merged[sym]
            primary = next(idx for idx in priority_order if idx in info["indices"])
            all_idx = "|".join(sorted(info["indices"], key=priority_order.index))
            writer.writerow([sym, info["company_name"], info["series"], primary, all_idx])

    logger.info(
        "Wrote %d unique symbols to %s", len(merged), config.SYMBOLS_MASTER_FILE
    )
    return config.SYMBOLS_MASTER_FILE


if __name__ == "__main__":
    build_symbol_universe()
