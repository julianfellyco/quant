"""backtester/data/benchmark.py — Benchmark return fetching and caching.

Fetches daily log returns for a benchmark ticker (default: SPY) using
yfinance and caches to parquet.  The cache lives alongside price data in
backtester/data_cache/.
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent / "data_cache"
_CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_BENCHMARK = "SPY"


def get_benchmark_returns(
    start_date: str,
    end_date: str,
    ticker: str = DEFAULT_BENCHMARK,
) -> pl.Series:
    """Fetch benchmark daily log returns, cached to parquet.

    Args:
        start_date: ISO date string "YYYY-MM-DD"
        end_date:   ISO date string "YYYY-MM-DD"
        ticker:     benchmark ticker (default "SPY")

    Returns:
        pl.Series of log returns (Float64), unnamed.
        Returns empty Series on failure.
    """
    cache_path = _CACHE_DIR / f"{ticker}_benchmark.parquet"

    # Try cache first
    if cache_path.exists():
        try:
            cached = pl.read_parquet(cache_path)
            mask = (
                (pl.col("date").cast(pl.Utf8) >= start_date) &
                (pl.col("date").cast(pl.Utf8) <= end_date)
            )
            filtered = cached.filter(mask)
            if not filtered.is_empty():
                return filtered["log_return"]
        except Exception as exc:
            logger.debug("Benchmark cache read failed: %s", exc)

    # Fetch from yfinance
    try:
        import yfinance as yf
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )

        if raw.empty:
            logger.warning("Benchmark fetch returned empty for %s", ticker)
            return pl.Series([], dtype=pl.Float64)

        # Flatten MultiIndex if present
        if hasattr(raw.columns, "levels"):
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

        import numpy as np

        closes = raw["Close"].dropna().values
        log_rets = np.log(closes[1:] / closes[:-1])
        dates = raw.index[1:]

        df = pl.DataFrame({
            "date":       [str(d)[:10] for d in dates],
            "log_return": log_rets.tolist(),
        })

        # Cache to parquet
        try:
            df.write_parquet(cache_path)
        except Exception as exc:
            logger.debug("Benchmark cache write failed: %s", exc)

        return df["log_return"]

    except Exception as exc:
        logger.warning("Benchmark fetch failed for %s: %s", ticker, exc)
        return pl.Series([], dtype=pl.Float64)


def get_benchmark_stats(
    start_date: str,
    end_date: str,
    ticker: str = DEFAULT_BENCHMARK,
) -> dict:
    """Return summary stats for the benchmark over the requested period.

    Returns dict with: ticker, total_return, sharpe (or None on failure).
    """
    import math

    rets = get_benchmark_returns(start_date, end_date, ticker)
    if rets.is_empty():
        return {"ticker": ticker, "total_return": None, "sharpe": None}

    total_return = float((rets.sum().exp() - 1))
    mu = float(rets.mean() or 0.0)
    sigma = float(rets.std() or 1e-9)
    rfr_daily = math.log(1 + 0.05) / 252
    sharpe = ((mu - rfr_daily) / sigma) * math.sqrt(252) if sigma > 1e-8 else None

    return {
        "ticker":       ticker,
        "total_return": round(total_return, 4) if not math.isnan(total_return) else None,
        "sharpe":       round(sharpe, 4) if sharpe and not math.isnan(sharpe) else None,
    }
