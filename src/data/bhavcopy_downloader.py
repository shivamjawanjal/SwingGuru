"""
Phase 1 / Step 2 — Download OHLCV (6 years of daily bhavcopy)

NSE publishes one "bhavcopy" zip per trading day containing EOD OHLCV
for every listed symbol. We need ~1500 trading days over 6 years, so
this is the step where concurrency actually matters: each day's file
is downloaded independently, so we fan them out across a thread pool
(I/O-bound — threads are the right tool, not processes) while a shared
rate limiter (in http_client) keeps us from tripping NSE's abuse
detection.

Files are cached to data/raw/bhavcopy/<year>/<yyyymmdd>.zip so re-runs
skip anything already downloaded.
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import List, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config
from src.utils import http_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bhavcopy_downloader")


def _trading_day_candidates(start: date, end: date) -> List[date]:
    """
    All Mon-Fri dates in range. This intentionally includes NSE holidays —
    those requests will just 404 and get skipped. Simpler and more robust
    than maintaining a holiday calendar that goes stale.
    """
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # 0=Mon ... 4=Fri
            days.append(d)
        d += timedelta(days=1)
    return days


def _target_path(d: date) -> Path:
    year_dir = config.BHAVCOPY_RAW_DIR / str(d.year)
    year_dir.mkdir(parents=True, exist_ok=True)
    return year_dir / f"{d.strftime('%Y%m%d')}.csv.zip"


def _download_one_day(d: date) -> Tuple[date, bool, str]:
    """Returns (date, success, message)."""
    dest = _target_path(d)
    if dest.exists() and dest.stat().st_size > 0:
        return d, True, "cached"

    url = config.BHAVCOPY_URL_TEMPLATE.format(yyyymmdd=d.strftime("%Y%m%d"))
    resp = http_client.get(url)
    if resp is None:
        return d, False, "not available (holiday or fetch failed)"

    dest.write_bytes(resp.content)
    return d, True, "downloaded"


def download_bhavcopy_range(start: date = None, end: date = None) -> dict:
    """
    Downloads all trading-day bhavcopy files in [start, end] concurrently.
    Returns a summary dict with counts.
    """
    start = start or config.START_DATE
    end = end or config.END_DATE

    candidates = _trading_day_candidates(start, end)
    logger.info(
        "Fetching bhavcopy for %d candidate weekdays between %s and %s "
        "(max %d concurrent downloads)",
        len(candidates), start, end, config.MAX_CONCURRENT_DOWNLOADS,
    )

    ok_downloaded, ok_cached, failed = 0, 0, 0
    failures = []

    with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_DOWNLOADS) as pool:
        futures = {pool.submit(_download_one_day, d): d for d in candidates}
        for i, future in enumerate(as_completed(futures), 1):
            d, success, msg = future.result()
            if success and msg == "downloaded":
                ok_downloaded += 1
            elif success and msg == "cached":
                ok_cached += 1
            else:
                failed += 1
                failures.append((d, msg))

            if i % 100 == 0 or i == len(candidates):
                logger.info(
                    "Progress: %d/%d checked | %d downloaded | %d cached | %d skipped",
                    i, len(candidates), ok_downloaded, ok_cached, failed,
                )

    logger.info(
        "Done. Downloaded %d new, %d already cached, %d skipped (holidays/failures).",
        ok_downloaded, ok_cached, failed,
    )
    return {
        "downloaded": ok_downloaded,
        "cached": ok_cached,
        "skipped": failed,
        "skipped_dates_sample": failures[:10],
    }


if __name__ == "__main__":
    download_bhavcopy_range()
