"""
data/downloader.py — Correct yfinance download for hourly OHLCV data.

Bugs fixed vs. the original snippet
-------------------------------------
1. 'Adj Close' KeyError
   yfinance >= 0.2 sets auto_adjust=True by default. The adjustment is folded
   into 'Close'; 'Adj Close' no longer exists. Fix: keep auto_adjust=True and
   access 'Close', which is already split-and-dividend-adjusted.

2. Only 2 columns saved (timestamp + price)
   The DataHandler needs open, high, low, close, volume to compute ATR (used
   by the cost model) and avg_volume_20 (for volume-liquidity adjustment).
   Saving only 'price' breaks both features silently — no error at download
   time, only at backtest time when column lookups fail.

3. Wrong file name and location
   '{ticker}_data.parquet' lands in CWD. DataHandler reads from
   data_cache/{ticker}_{granularity}.parquet. Renamed accordingly.

4. Hourly granularity not wired in DataHandler
   Handler only knew DAILY / MINUTE. Added Granularity.HOUR with
   ann_factor = 252 * HOURS_PER_DAY = 1,764 bars/year.

5. Timestamp column name collision on sub-daily data
   yfinance names the index 'Datetime' (not 'Date') for intervals < 1d.
   The original reset_index() + manual column rename silently discarded all
   five OHLCV columns and kept only the index as 'timestamp'.

MultiIndex handling (yfinance >= 0.2)
--------------------------------------
Downloading multiple tickers returns a two-level column index:
    level 0 (Price): Open, High, Low, Close, Volume
    level 1 (Ticker): PFE, NVO

    data['Close']['PFE']  → Series for PFE close
    data[('Close','PFE')] → same thing

We iterate tickers explicitly and call yf.Ticker(t).history() per ticker
to get a simple single-level DataFrame — avoids the MultiIndex complexity
entirely and makes the schema deterministic.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).parent.parent / "data_cache"
DATA_DIR.mkdir(exist_ok=True)

# NYSE regular session: 9:30–16:00 → 7 hourly bars
# (9:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30)
HOURS_PER_TRADING_DAY = 7
TRADING_DAYS_PER_YEAR = 252
# Annualisation factor for hourly Sharpe / Sortino
ANN_FACTOR_HOUR = TRADING_DAYS_PER_YEAR * HOURS_PER_TRADING_DAY   # 1,764


def download_pharma_data(
    tickers:  list[str] = None,
    period:   str       = "1y",
    interval: str       = "1h",
    force:    bool      = False,
) -> dict[str, pl.DataFrame]:
    """
    Download hourly OHLCV for each ticker and persist to Parquet.

    Args:
        tickers:  list of ticker symbols (default: ["PFE", "NVO"])
        period:   yfinance period string ("1y", "2y", "6mo", …)
                  yfinance supports hourly data up to 730 days back.
        interval: bar size ("1h", "30m", "1d", …)
        force:    re-download even if the parquet file already exists

    Returns:
        Dict mapping ticker → Polars DataFrame (full OHLCV schema).

    Parquet schema written to data_cache/{ticker}_hour.parquet:
        timestamp  Datetime[us, UTC]
        open       Float64
        high       Float64
        low        Float64
        close      Float64     ← split & dividend adjusted (auto_adjust=True)
        volume     Int64
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("pip install yfinance") from exc

    if tickers is None:
        tickers = ["PFE", "NVO"]

    results: dict[str, pl.DataFrame] = {}

    for ticker in tickers:
        suffix   = interval.replace("m", "min").replace("h", "hour")
        out_path = DATA_DIR / f"{ticker}_{suffix}.parquet"

        if out_path.exists() and not force:
            print(f"[{ticker}] Cache hit → {out_path.name}")
            results[ticker] = pl.read_parquet(out_path)
            continue

        print(f"[{ticker}] Downloading {interval} bars ({period})...")

        # ── Bug 1 fix: auto_adjust=True is default; use 'Close' not 'Adj Close'
        # ── Bug 5 fix: use Ticker.history() → single-level columns, no MultiIndex
        t_obj = yf.Ticker(ticker)
        raw   = t_obj.history(period=period, interval=interval, auto_adjust=True)

        if raw.empty:
            raise ValueError(
                f"yfinance returned no data for {ticker} "
                f"(period={period}, interval={interval})"
            )

        # ── Bug 5 fix: index is named 'Datetime' for sub-daily, normalise it
        raw.index.name = "timestamp"
        raw = raw.reset_index()

        # Lowercase all column names for consistency
        raw.columns = [c.lower() for c in raw.columns]

        # ── Bug 2 fix: keep full OHLCV, not just close
        keep = [c for c in ("timestamp", "open", "high", "low", "close", "volume")
                if c in raw.columns]
        missing = {"open", "high", "low", "close", "volume"} - set(keep)
        if missing:
            raise ValueError(
                f"yfinance response missing expected columns: {missing}. "
                f"Got: {list(raw.columns)}"
            )
        raw = raw[keep]

        # Convert to Polars with explicit dtypes
        df = pl.from_pandas(raw).with_columns(
            # Preserve timezone info if present; cast to UTC Datetime[us]
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
        ).sort("timestamp")

        # ── Bug 3 fix: write to data_cache/ with the correct naming convention
        df.write_parquet(out_path)
        print(f"[{ticker}] Saved {len(df):,} bars → {out_path.name}")

        results[ticker] = df

    return results


def inspect(df: pl.DataFrame, ticker: str) -> None:
    """Print a quick sanity-check summary of a downloaded frame."""
    print(f"\n{'─'*50}")
    print(f"  {ticker}  │  {len(df):,} bars")
    print(f"  {df['timestamp'][0]}  →  {df['timestamp'][-1]}")
    print(f"  Close range: ${float(df['close'].min()):.2f} – ${float(df['close'].max()):.2f}")
    print(f"  Null counts: {dict(zip(df.columns, df.null_count().row(0)))}")
    print(f"  Schema: {dict(df.schema)}")


# --------------------------------------------------------------------------- #
# Run as a standalone script                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    frames = download_pharma_data(force=False)
    for ticker, df in frames.items():
        inspect(df, ticker)

    print(f"\nAnnualisation factor for hourly bars: {ANN_FACTOR_HOUR} bars/year")
    print("Pass this to VectorizedEngine(ann_factor=1764) when backtesting hourly data.")
