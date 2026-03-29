"""
data/universe_fetcher.py — Parallel data fetcher for 500+ ticker universes.

Threading approach
------------------
yfinance is built on requests (synchronous HTTP) and has no async interface.
Wrapping each download in asyncio.run_in_executor would work but adds overhead
and complexity for no benefit — the bottleneck is network I/O, not the event
loop. ThreadPoolExecutor is the correct tool here:

  1. Each worker thread calls yf.Ticker(ticker).history() independently.
  2. The GIL is released during the underlying socket reads, so true parallel
     I/O occurs across all workers.
  3. max_workers=20 is a safe default for a typical home/office connection;
     lower it to 8-10 if you hit yfinance rate limits.
  4. Failed tickers are caught per-worker and logged; they do NOT cancel other
     downloads or raise in the calling thread.

Cache convention
----------------
Parquet files land at data_cache/{ticker}_{granularity}.parquet, matching the
naming used by downloader.py and DataHandler._load_raw().

  granularity="daily"  → data_cache/AAPL_daily.parquet
  granularity="hour"   → data_cache/AAPL_hour.parquet
  granularity="minute" → data_cache/AAPL_minute.parquet

Date filtering
--------------
start/end are ISO date strings ("2022-01-01"). After loading from cache or
yfinance, we filter the DataFrame to [start, end] so callers always get a
time-bounded slice regardless of what is stored on disk.
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data_cache"
DATA_DIR.mkdir(exist_ok=True)

# ── Granularity mappings ────────────────────────────────────────────────────── #

_INTERVAL_MAP: dict[str, str] = {
    "daily":  "1d",
    "hour":   "1h",
    "minute": "1m",
}

# yfinance hard limits: hourly data only available for last 730 days,
# minute data only for the last 7 days.
_PERIOD_MAP: dict[str, str] = {
    "daily":  "5y",
    "hour":   "730d",
    "minute": "7d",
}


# --------------------------------------------------------------------------- #
# Private: single-ticker worker                                                 #
# --------------------------------------------------------------------------- #

def _fetch_one(
    ticker: str,
    start: str,
    end: str,
    granularity: str,
    force: bool,
    total: int,
    counter: list[int],   # mutable counter shared across threads (GIL-safe for int ops)
) -> tuple[str, Optional[pl.DataFrame]]:
    """
    Download one ticker and write/read parquet cache.

    Returns (ticker, DataFrame) on success or (ticker, None) on failure.
    The counter list holds a single integer; we increment it for progress display.
    Thread-safe because CPython's GIL makes simple int increments atomic.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("pip install yfinance") from exc

    parquet_path = DATA_DIR / f"{ticker}_{granularity}.parquet"

    # ── Cache hit ──────────────────────────────────────────────────────── #
    if parquet_path.exists() and not force:
        try:
            df = pl.read_parquet(parquet_path)
            df = _filter_dates(df, start, end)
            counter[0] += 1
            print(f"[OK] {ticker:<8}  {counter[0]}/{total}  (cache)")
            return ticker, df
        except Exception as exc:
            warnings.warn(f"[WARN] {ticker}: cache read failed ({exc}), re-downloading")

    # ── Download ───────────────────────────────────────────────────────── #
    try:
        interval = _INTERVAL_MAP[granularity]
        period   = _PERIOD_MAP[granularity]

        t_obj = yf.Ticker(ticker)
        # suppress yfinance progress noise in threaded context
        raw = t_obj.history(period=period, interval=interval, auto_adjust=True)

        if raw is None or raw.empty:
            raise ValueError(f"yfinance returned empty DataFrame for {ticker}")

        # Normalise index name (yfinance uses 'Datetime' for sub-daily, 'Date' for daily)
        raw.index.name = "timestamp"
        raw = raw.reset_index()
        raw.columns = [str(c).lower() for c in raw.columns]

        keep = [c for c in ("timestamp", "open", "high", "low", "close", "volume")
                if c in raw.columns]
        raw = raw[keep]

        df = pl.from_pandas(raw)

        # Cast timestamp to Datetime[us] (strip tz if present)
        if df["timestamp"].dtype != pl.Datetime("us"):
            try:
                df = df.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
                    .dt.convert_time_zone("UTC")
                    .dt.replace_time_zone(None)
                    .alias("timestamp")
                )
            except Exception:
                df = df.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("us")).alias("timestamp")
                )

        # Cast OHLCV columns
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64))
        if "volume" in df.columns:
            df = df.with_columns(pl.col("volume").cast(pl.Int64))

        df = df.sort("timestamp")

        # Write full history to cache
        df.write_parquet(parquet_path)

        # Filter to requested window
        df = _filter_dates(df, start, end)

        counter[0] += 1
        print(f"[OK] {ticker:<8}  {counter[0]}/{total}")
        return ticker, df

    except Exception as exc:
        counter[0] += 1
        print(f"[FAIL] {ticker:<8}  {counter[0]}/{total}  err={exc}")
        logger.warning("fetch_universe: failed ticker %s: %s", ticker, exc)
        return ticker, None


def _filter_dates(df: pl.DataFrame, start: str, end: str) -> pl.DataFrame:
    """Filter DataFrame to [start, end] using ISO date strings."""
    ts_col = df["timestamp"]
    # Cast to Date for comparison regardless of Datetime precision
    return df.filter(
        pl.col("timestamp").cast(pl.Date).is_between(
            pl.lit(start).str.to_date("%Y-%m-%d"),
            pl.lit(end).str.to_date("%Y-%m-%d"),
        )
    )


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #

def fetch_universe(
    tickers:     list[str],
    start:       str,
    end:         str,
    granularity: str = "daily",
    max_workers: int = 20,
    force:       bool = False,
) -> dict[str, pl.DataFrame]:
    """
    Download OHLCV data for a universe of tickers in parallel using threads.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols, e.g. ["AAPL", "MSFT", "GOOGL", ...].
    start : str
        Start date as ISO string, e.g. "2022-01-01".
    end : str
        End date as ISO string, e.g. "2024-12-31".
    granularity : str
        One of "daily", "hour", "minute". Maps to yfinance interval internally.
        yfinance hard limits: hourly ≤ 730 days back, minute ≤ 7 days back.
    max_workers : int
        ThreadPoolExecutor concurrency level. Default 20. Lower if rate-limited.
    force : bool
        Re-download even if a cached parquet file exists. Default False.

    Returns
    -------
    dict[str, pl.DataFrame]
        Mapping of ticker → Polars DataFrame (OHLCV + timestamp). Only
        successfully downloaded tickers are included; failed tickers are
        logged and excluded silently.

    Notes
    -----
    Threading is used (not asyncio) because yfinance uses the requests library
    (synchronous HTTP). The GIL is released during socket I/O, so true
    parallel network fetches occur across all worker threads.
    """
    if granularity not in _INTERVAL_MAP:
        raise ValueError(
            f"granularity must be one of {list(_INTERVAL_MAP.keys())}, got {granularity!r}"
        )

    total   = len(tickers)
    counter = [0]   # mutable single-element list for thread-safe counter

    results: dict[str, pl.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_one, ticker, start, end, granularity, force, total, counter): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker, df = future.result()
            if df is not None:
                results[ticker] = df

    print(f"\nfetch_universe complete: {len(results)}/{total} tickers OK")
    return results


def fetch_universe_long(
    tickers:     list[str],
    start:       str,
    end:         str,
    granularity: str = "daily",
    max_workers: int = 20,
    force:       bool = False,
) -> pl.DataFrame:
    """
    Download universe data and return a single long-format DataFrame.

    Calls fetch_universe() then vertically concatenates all per-ticker frames,
    adding a "ticker" string column. Rows are sorted by (ticker, timestamp).

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols.
    start : str
        Start date as ISO string, e.g. "2022-01-01".
    end : str
        End date as ISO string, e.g. "2024-12-31".
    granularity : str
        "daily", "hour", or "minute".
    max_workers : int
        Thread pool size. Default 20.
    force : bool
        Bypass cache and re-download. Default False.

    Returns
    -------
    pl.DataFrame
        Long-format DataFrame with columns:
            ticker      Utf8
            timestamp   Datetime[us]
            open        Float64
            high        Float64
            low         Float64
            close       Float64
            volume      Int64
        Sorted by (ticker, timestamp).
    """
    frames_map = fetch_universe(
        tickers=tickers,
        start=start,
        end=end,
        granularity=granularity,
        max_workers=max_workers,
        force=force,
    )

    if not frames_map:
        return pl.DataFrame(schema={
            "ticker": pl.Utf8,
            "timestamp": pl.Datetime("us"),
        })

    labelled = [
        df.with_columns(pl.lit(ticker).alias("ticker"))
        for ticker, df in frames_map.items()
    ]

    long_df = pl.concat(labelled, how="diagonal").sort(["ticker", "timestamp"])
    return long_df
