"""
data/metadata.py — Dynamic ticker metadata via yfinance.

Replaces the hardcoded TICKER_META dict in handler.py with live-fetched
ADV, price, sector, exchange, and market cap for any ticker.

Cache strategy: results are written to data_cache/metadata.json as a flat
dict {ticker: {...}} so the next run skips the network call.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #

DATA_DIR = Path(__file__).parent.parent / "data_cache"
DATA_DIR.mkdir(exist_ok=True)

CACHE_PATH = DATA_DIR / "metadata.json"

# --------------------------------------------------------------------------- #
# Static fallback (copied from handler.py's TICKER_META)                       #
# --------------------------------------------------------------------------- #

STATIC_META: dict[str, dict] = {
    "PFE": {
        "name": "Pfizer Inc.",
        "exchange": "NYSE",
        "avg_daily_volume": 28_000_000,
        "approx_price_2024": 28.0,
        "approx_price": 28.0,
        "market_cap": None,
        "sector": "Healthcare",
    },
    "NVO": {
        "name": "Novo Nordisk ADR",
        "exchange": "NYSE",
        "avg_daily_volume": 3_500_000,
        "approx_price_2024": 110.0,
        "approx_price": 110.0,
        "market_cap": None,
        "sector": "Healthcare",
    },
}

# Module-level lock for thread-safe cache writes
_cache_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Live fetch                                                                    #
# --------------------------------------------------------------------------- #

def fetch_live_metadata(ticker: str) -> dict:
    """
    Fetch live metadata for a ticker using yfinance fast_info.

    Returns a dict with keys:
        avg_daily_volume, approx_price, approx_price_2024, market_cap,
        sector, exchange, name
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("pip install yfinance") from exc

    t = yf.Ticker(ticker)
    fi = t.fast_info

    # avg_daily_volume
    try:
        avg_daily_volume = int(fi.three_month_average_volume or 1_000_000)
    except Exception:
        avg_daily_volume = 1_000_000

    # approx_price
    try:
        approx_price = float(fi.last_price or 50.0)
    except Exception:
        approx_price = 50.0

    # market_cap
    try:
        market_cap = fi.market_cap
        if market_cap is not None:
            market_cap = float(market_cap)
    except Exception:
        market_cap = None

    # sector — requires .info (slower), wrap in try/except
    try:
        sector = t.info.get("sector", "Unknown") or "Unknown"
    except Exception:
        sector = "Unknown"

    # exchange
    try:
        exchange = fi.exchange or "NYSE"
    except Exception:
        exchange = "NYSE"

    # name
    try:
        name = fi.quote_type or ticker
    except Exception:
        name = ticker

    return {
        "name": name,
        "exchange": exchange,
        "avg_daily_volume": avg_daily_volume,
        "approx_price": approx_price,
        "approx_price_2024": approx_price,   # alias for backward compat
        "market_cap": market_cap,
        "sector": sector,
    }


# --------------------------------------------------------------------------- #
# Cache helpers                                                                 #
# --------------------------------------------------------------------------- #

def _load_cache() -> dict[str, dict]:
    """Load the metadata JSON cache from disk. Returns {} if missing/corrupt."""
    try:
        if CACHE_PATH.exists():
            with open(CACHE_PATH, "r") as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}


def _save_cache(cache: dict[str, dict]) -> None:
    """Write the metadata cache dict to disk (caller must hold _cache_lock)."""
    try:
        with open(CACHE_PATH, "w") as fh:
            json.dump(cache, fh, indent=2, default=str)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #

def get_metadata(ticker: str, use_cache: bool = True) -> dict:
    """
    Return metadata for a single ticker.

    Resolution order:
      1. STATIC_META (hardcoded PFE/NVO — no network call ever)
      2. data_cache/metadata.json (on-disk JSON cache)
      3. fetch_live_metadata() — live yfinance call; result is written to cache

    Thread-safe: cache writes are protected by a module-level threading.Lock.
    """
    # 1. Static fallback — always preferred for known tickers
    if ticker in STATIC_META:
        return STATIC_META[ticker]

    if use_cache:
        # 2. Check JSON cache (read without lock — reads are safe)
        cache = _load_cache()
        if ticker in cache:
            return cache[ticker]

    # 3. Live fetch
    meta = fetch_live_metadata(ticker)

    if use_cache:
        with _cache_lock:
            # Reload inside lock to avoid overwriting concurrent writes
            cache = _load_cache()
            cache[ticker] = meta
            _save_cache(cache)

    return meta


def get_metadata_batch(
    tickers: list[str],
    max_workers: int = 10,
) -> dict[str, dict]:
    """
    Fetch metadata for multiple tickers in parallel using ThreadPoolExecutor.

    Returns a dict keyed by ticker. Static tickers are resolved without network
    calls; unknown tickers are fetched concurrently.
    """
    results: dict[str, dict] = {}

    # Resolve static tickers immediately
    static_hits = [t for t in tickers if t in STATIC_META]
    dynamic_tickers = [t for t in tickers if t not in STATIC_META]

    for t in static_hits:
        results[t] = STATIC_META[t]

    if not dynamic_tickers:
        return results

    # Load cache once outside the thread pool to avoid redundant I/O
    cache = _load_cache()
    still_needed: list[str] = []
    for t in dynamic_tickers:
        if t in cache:
            results[t] = cache[t]
        else:
            still_needed.append(t)

    if not still_needed:
        return results

    # Fetch remaining tickers in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_ticker = {
            pool.submit(fetch_live_metadata, t): t
            for t in still_needed
        }
        for future in as_completed(future_to_ticker):
            t = future_to_ticker[future]
            try:
                meta = future.result()
                results[t] = meta
            except Exception:
                # Provide a safe default on failure
                results[t] = {
                    "name": t,
                    "exchange": "NYSE",
                    "avg_daily_volume": 1_000_000,
                    "approx_price": 50.0,
                    "approx_price_2024": 50.0,
                    "market_cap": None,
                    "sector": "Unknown",
                }

    # Write all new results to cache in a single lock acquisition
    with _cache_lock:
        updated_cache = _load_cache()
        for t in still_needed:
            if t in results:
                updated_cache[t] = results[t]
        _save_cache(updated_cache)

    return results
