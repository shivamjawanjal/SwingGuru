"""
Shared HTTP client for talking to NSE's servers.

NSE requires a warmed-up session (visit the homepage first to pick up
cookies) and browser-like headers, or it responds with 401/403. This
module centralizes that so every downloader reuses one working session
instead of each one reinventing cookie handling.

Thread-safe: each thread gets its own requests.Session via thread-local
storage, but they all share the same rate limiter so we never hammer
NSE faster than MIN_DELAY_BETWEEN_REQUESTS regardless of how many
worker threads are running concurrently.
"""

import threading
import time
import logging
from typing import Optional

import requests

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config

logger = logging.getLogger("nse_http_client")

_thread_local = threading.local()
_rate_lock = threading.Lock()
_last_request_time = [0.0]  # mutable container so we can update inside lock


def _throttle():
    """Ensure at least MIN_DELAY_BETWEEN_REQUESTS between any two requests,
    across all threads."""
    with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_request_time[0]
        wait = config.MIN_DELAY_BETWEEN_REQUESTS - elapsed
        if wait > 0:
            time.sleep(wait)
        _last_request_time[0] = time.monotonic()


def _get_session() -> requests.Session:
    """Return a warmed-up session unique to the current thread."""
    if getattr(_thread_local, "session", None) is not None:
        return _thread_local.session

    session = requests.Session()
    session.headers.update(config.NSE_HEADERS)
    try:
        # Hitting the homepage first sets the cookies NSE checks for on
        # archive/API requests. If this fails we still return the session;
        # some archive endpoints work without it.
        session.get(config.NSE_HOME_URL, timeout=config.REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        logger.warning("Could not warm up NSE session: %s", exc)

    _thread_local.session = session
    return session


def get(url: str, **kwargs) -> Optional[requests.Response]:
    """
    GET a URL through the shared, rate-limited, retrying NSE session.
    Returns None (after logging) if all retries are exhausted.
    """
    session = _get_session()
    last_exc = None

    for attempt in range(1, config.MAX_RETRIES + 1):
        _throttle()
        try:
            resp = session.get(
                url, timeout=config.REQUEST_TIMEOUT_SECONDS, **kwargs
            )
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                # Not found is often expected (e.g. weekends/holidays have
                # no bhavcopy) — don't retry, just report cleanly.
                logger.debug("404 for %s", url)
                return None
            logger.warning(
                "Non-200 (%s) for %s [attempt %d/%d]",
                resp.status_code, url, attempt, config.MAX_RETRIES,
            )
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning(
                "Request error for %s [attempt %d/%d]: %s",
                url, attempt, config.MAX_RETRIES, exc,
            )
            # Session may have gone stale (expired cookies) — force a
            # fresh one on the next attempt.
            _thread_local.session = None
            session = _get_session()

        if attempt < config.MAX_RETRIES:
            time.sleep(config.RETRY_BACKOFF_SECONDS * attempt)

    if last_exc:
        logger.error("Giving up on %s after %d attempts: %s", url, config.MAX_RETRIES, last_exc)
    else:
        logger.error("Giving up on %s after %d attempts", url, config.MAX_RETRIES)
    return None
