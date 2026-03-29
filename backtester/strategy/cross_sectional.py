"""
strategy/cross_sectional.py — Universe-level signals for cross-sectional strategies.

Unlike single-stock signals, cross-sectional signals rank ALL stocks at each
bar and generate positions based on relative rank rather than absolute thresholds.

This is how institutional momentum strategies actually work: you do not say
"buy NVO because it's up 10%", you say "buy the top 20% of stocks by trailing
momentum across the S&P 500".
"""

from __future__ import annotations

import polars as pl


# --------------------------------------------------------------------------- #
# Universe return features                                                      #
# --------------------------------------------------------------------------- #

def compute_universe_returns(long_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add log_return and cum_return_60 columns to a long-format universe DataFrame.

    Parameters
    ----------
    long_df : pl.DataFrame
        Long-format DataFrame with columns: ticker, timestamp, close.

    Returns
    -------
    pl.DataFrame
        Augmented DataFrame with additional columns:
            log_return    — ln(close / close.shift(1)) per ticker
            cum_return_60 — sum of last 60 log returns (skip last 5) per ticker
    """
    df = long_df.sort(["ticker", "timestamp"])

    # log_return: PiT-safe (uses previous close)
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ticker"))
        .log()
        .alias("log_return")
    )

    # cum_return_60: skip-5 momentum window (shift(5) then rolling_sum(55))
    df = df.with_columns(
        pl.col("log_return")
        .shift(5)
        .over("ticker")
        .rolling_sum(window_size=55)
        .over("ticker")
        .alias("cum_return_60")
    )

    return df


# --------------------------------------------------------------------------- #
# Cross-sectional momentum rank signal                                          #
# --------------------------------------------------------------------------- #

def momentum_rank_signal(
    long_df: pl.DataFrame,
    lookback: int   = 60,
    skip:     int   = 5,
    top_pct:  float = 0.2,
    bottom_pct: float = 0.2,
) -> pl.DataFrame:
    """
    Cross-sectional momentum signal: long top performers, short bottom.

    At each timestamp bar, compute each ticker's trailing `lookback`-bar return
    (skipping last `skip` bars), rank tickers by that return, and assign:
        +1  → top `top_pct` fraction (winners)
        -1  → bottom `bottom_pct` fraction (losers)
         0  → middle (neutral)

    Signal is shifted 1 bar forward (point-in-time guarantee) before returning.

    Parameters
    ----------
    long_df : pl.DataFrame
        Long-format DataFrame with columns: ticker, timestamp, close.
    lookback : int
        Total lookback window in bars. Default 60.
    skip : int
        Number of recent bars to skip (short-term reversal avoidance). Default 5.
    top_pct : float
        Fraction of universe to go long. Default 0.2 (top 20%).
    bottom_pct : float
        Fraction of universe to go short. Default 0.2 (bottom 20%).

    Returns
    -------
    pl.DataFrame
        Long-format DataFrame with columns: ticker, timestamp, signal (Float64).
    """
    df = long_df.select(["ticker", "timestamp", "close"]).sort(["ticker", "timestamp"])

    # Compute log returns per ticker
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ticker"))
        .log()
        .alias("log_return")
    )

    # trailing return: shift(skip) then rolling_sum(lookback - skip)
    roll_window = max(1, lookback - skip)
    df = df.with_columns(
        pl.col("log_return")
        .shift(skip)
        .over("ticker")
        .rolling_sum(window_size=roll_window)
        .over("ticker")
        .alias("trailing_return")
    )

    # Cross-sectional rank at each timestamp (min-rank, 1-based)
    df = df.with_columns(
        pl.col("trailing_return")
        .rank(method="min")
        .over("timestamp")
        .alias("cs_rank")
    )

    # Count valid (non-null) tickers per timestamp for relative rank
    df = df.with_columns(
        pl.col("trailing_return")
        .count()
        .over("timestamp")
        .alias("n_valid")
    )

    # Assign signal based on rank percentile.
    # Use strict inequality on the upper boundary (> rather than >=) to ensure
    # n_long == n_short when top_pct == bottom_pct, keeping the portfolio
    # market-neutral at every bar.
    df = df.with_columns(
        pl.when(pl.col("trailing_return").is_null())
        .then(pl.lit(0.0))
        .when((pl.col("cs_rank") / pl.col("n_valid")) > (1.0 - top_pct))
        .then(pl.lit(1.0))
        .when((pl.col("cs_rank") / pl.col("n_valid")) <= bottom_pct)
        .then(pl.lit(-1.0))
        .otherwise(pl.lit(0.0))
        .cast(pl.Float64)
        .alias("raw_signal")
    )

    # Point-in-time: shift signal forward 1 bar per ticker
    df = df.with_columns(
        pl.col("raw_signal")
        .shift(1)
        .over("ticker")
        .fill_null(0.0)
        .alias("signal")
    )

    return df.select(["ticker", "timestamp", "signal"])


# --------------------------------------------------------------------------- #
# Cross-sectional mean reversion rank signal                                    #
# --------------------------------------------------------------------------- #

def mean_reversion_rank_signal(
    long_df: pl.DataFrame,
    zscore_window: int   = 20,
    entry_pct:     float = 0.2,
) -> pl.DataFrame:
    """
    Cross-sectional mean reversion signal: short overbought, long oversold.

    For each ticker, compute a rolling z-score of log returns over `zscore_window`
    bars. Then rank by z-score cross-sectionally at each timestamp:
        -1  → top `entry_pct` fraction (most overbought — short)
        +1  → bottom `entry_pct` fraction (most oversold — long)
         0  → middle

    Signal is shifted 1 bar forward (point-in-time guarantee) before returning.

    Parameters
    ----------
    long_df : pl.DataFrame
        Long-format DataFrame with columns: ticker, timestamp, close.
    zscore_window : int
        Rolling window for z-score computation. Default 20.
    entry_pct : float
        Fraction of universe at each extreme to trade. Default 0.2 (top/bottom 20%).

    Returns
    -------
    pl.DataFrame
        Long-format DataFrame with columns: ticker, timestamp, signal (Float64).
    """
    df = long_df.select(["ticker", "timestamp", "close"]).sort(["ticker", "timestamp"])

    # Compute log returns per ticker
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ticker"))
        .log()
        .alias("log_return")
    )

    # Rolling z-score per ticker (PiT: use shift(1) before rolling stats)
    lagged = pl.col("log_return").shift(1).over("ticker")

    df = df.with_columns(
        lagged.rolling_mean(window_size=zscore_window).over("ticker").alias("_roll_mean"),
        lagged.rolling_std(window_size=zscore_window).over("ticker").alias("_roll_std"),
    ).with_columns(
        (
            (lagged - pl.col("_roll_mean"))
            / pl.col("_roll_std").clip(lower_bound=1e-9)
        )
        .over("ticker")
        .alias("zscore")
    ).drop(["_roll_mean", "_roll_std"])

    # Cross-sectional rank of z-score at each timestamp
    df = df.with_columns(
        pl.col("zscore")
        .rank(method="min")
        .over("timestamp")
        .alias("cs_rank")
    )

    df = df.with_columns(
        pl.col("zscore")
        .count()
        .over("timestamp")
        .alias("n_valid")
    )

    # Signal: short overbought (high z), long oversold (low z).
    # Use strict inequality on the upper boundary to ensure market neutrality.
    df = df.with_columns(
        pl.when(pl.col("zscore").is_null())
        .then(pl.lit(0.0))
        .when((pl.col("cs_rank") / pl.col("n_valid")) > (1.0 - entry_pct))
        .then(pl.lit(-1.0))   # overbought → short
        .when((pl.col("cs_rank") / pl.col("n_valid")) <= entry_pct)
        .then(pl.lit(1.0))    # oversold → long
        .otherwise(pl.lit(0.0))
        .cast(pl.Float64)
        .alias("raw_signal")
    )

    # Point-in-time: shift signal forward 1 bar per ticker
    df = df.with_columns(
        pl.col("raw_signal")
        .shift(1)
        .over("ticker")
        .fill_null(0.0)
        .alias("signal")
    )

    return df.select(["ticker", "timestamp", "signal"])
