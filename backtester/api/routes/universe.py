"""api/routes/universe.py — POST /api/universe

Runs a cross-sectional universe backtest:
  1. Resolve ticker list (screener preset OR custom list)
  2. Fetch OHLCV in parallel via fetch_universe_long()
  3. Add momentum + z-score features via compute_universe_returns()
  4. Run UniverseEngine with the chosen signal
  5. Return leaderboard + aggregate stats
"""
from __future__ import annotations

import datetime as dt

import polars as pl
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backtester.data.screener         import Screener
from backtester.data.universe_fetcher import fetch_universe_long
from backtester.engine.universe_engine import UniverseEngine
from backtester.strategy.cross_sectional import (
    compute_universe_returns,
    momentum_rank_signal,
    mean_reversion_rank_signal,
)
from backtester.strategy.signals import momentum_signal, mean_reversion_signal

from api.deps import clean_float, granularity_enum

router = APIRouter()


# ── Request / Response ────────────────────────────────────────────────────── #

class UniverseRequest(BaseModel):
    preset:          str   = "pharma"   # pharma | sp500 | nasdaq100 | custom
    custom_tickers:  list[str] = []
    strategy:        str   = "momentum_rank"  # momentum_rank | mean_reversion_rank | momentum | mean_reversion
    start_date:      str   = "2024-01-01"
    end_date:        str   = "2024-12-31"
    top_pct:         float = 0.2
    bottom_pct:      float = 0.2
    lookback:        int   = 60
    initial_capital: float = 100_000.0
    shares_per_unit: int   = 100
    risk_free_rate:  float = 0.05
    max_tickers:     int   = 50   # safety cap — keeps response time < 30s


class TickerResult(BaseModel):
    ticker:       str
    net_sharpe:   float | None
    sortino:      float | None
    total_return: float | None
    max_drawdown: float | None
    n_trades:     int
    total_cost_usd: float


class UniverseResponse(BaseModel):
    tickers_run:          int
    strategy:             str
    leaderboard:          list[TickerResult]   # sorted by net_sharpe desc
    mean_sharpe:          float | None
    median_sharpe:        float | None
    pct_positive_sharpe:  float | None
    best_ticker:          str | None
    worst_ticker:         str | None


# ── Helpers ───────────────────────────────────────────────────────────────── #

def _resolve_tickers(req: UniverseRequest) -> list[str]:
    if req.preset == "custom":
        tickers = [t.strip().upper() for t in req.custom_tickers if t.strip()]
        if not tickers:
            raise HTTPException(400, "custom_tickers is empty")
        return tickers[:req.max_tickers]
    if req.preset == "pharma":
        return Screener.pharma_large_cap()[:req.max_tickers]
    if req.preset == "sp500":
        try:
            return Screener.sp500()[:req.max_tickers]
        except Exception as e:
            raise HTTPException(503, f"Could not fetch S&P 500 list: {e}")
    if req.preset == "nasdaq100":
        try:
            return Screener.nasdaq100()[:req.max_tickers]
        except Exception as e:
            raise HTTPException(503, f"Could not fetch Nasdaq-100 list: {e}")
    raise HTTPException(400, f"Unknown preset '{req.preset}'. Use: pharma, sp500, nasdaq100, custom")


# ── Route ─────────────────────────────────────────────────────────────────── #

@router.post("/universe", response_model=UniverseResponse)
def run_universe(req: UniverseRequest) -> UniverseResponse:
    try:
        start = dt.date.fromisoformat(req.start_date)
        end   = dt.date.fromisoformat(req.end_date)
    except ValueError as e:
        raise HTTPException(400, f"Invalid date: {e}")

    tickers = _resolve_tickers(req)

    # ── Fetch OHLCV in parallel ──────────────────────────────────────────── #
    try:
        long_df = fetch_universe_long(
            tickers,
            start       = req.start_date,
            end         = req.end_date,
            granularity = "daily",
            max_workers = 20,
        )
    except Exception as e:
        raise HTTPException(400, f"Data fetch failed: {e}")

    if long_df.is_empty():
        raise HTTPException(400, "No data returned for any ticker in the requested range.")

    actual_tickers = long_df["ticker"].unique().to_list()

    # ── Add momentum / z-score features ─────────────────────────────────── #
    try:
        long_df = compute_universe_returns(long_df)
    except Exception as e:
        raise HTTPException(500, f"Feature computation failed: {e}")

    # ── Add columns needed by single-stock signals ───────────────────────── #
    # Ensure atr_14, avg_volume_20, zscore_20d, momentum_60_5 exist
    long_df = long_df.sort(["ticker", "timestamp"]).with_columns([
        # ATR-14 proxy: rolling std of log_return × close
        (pl.col("log_return").abs().rolling_mean(window_size=14, min_samples=1)
           .over("ticker") * pl.col("close")).alias("atr_14"),
        # avg_volume_20
        pl.col("volume").cast(pl.Float64)
          .rolling_mean(window_size=20, min_samples=1)
          .over("ticker").alias("avg_volume_20"),
        # zscore_20d
        ((pl.col("log_return") - pl.col("log_return").rolling_mean(window_size=20, min_samples=1).over("ticker"))
         / (pl.col("log_return").rolling_std(window_size=20, min_samples=1).over("ticker") + 1e-9))
        .alias("zscore_20d"),
        # momentum_60_5: sum of log returns from bar t-60 to t-5 (skip last 5)
        pl.col("log_return").shift(5).over("ticker")
          .rolling_sum(window_size=55, min_samples=1)
          .over("ticker").alias("momentum_60_5"),
        # is_event_window: all False (no pharma calendar for generic tickers)
        pl.lit(False).alias("is_event_window"),
    ])

    engine = UniverseEngine(
        initial_capital = req.initial_capital,
        shares_per_unit = req.shares_per_unit,
        risk_free_rate  = req.risk_free_rate,
        ann_factor      = 252,
    )

    # ── Pick signal function ─────────────────────────────────────────────── #
    strat = req.strategy
    try:
        if strat == "momentum_rank":
            sig_df = momentum_rank_signal(
                long_df,
                lookback   = req.lookback,
                top_pct    = req.top_pct,
                bottom_pct = req.bottom_pct,
            )
            # Merge signal back into long_df
            long_df = long_df.join(sig_df.select(["ticker", "timestamp", "signal"]),
                                   on=["ticker", "timestamp"], how="left")
            signal_fn = lambda df: df["signal"].fill_null(0.0)

        elif strat == "mean_reversion_rank":
            sig_df = mean_reversion_rank_signal(long_df, entry_pct=req.top_pct)
            long_df = long_df.join(sig_df.select(["ticker", "timestamp", "signal"]),
                                   on=["ticker", "timestamp"], how="left")
            signal_fn = lambda df: df["signal"].fill_null(0.0)

        elif strat == "momentum":
            signal_fn = momentum_signal

        elif strat == "mean_reversion":
            signal_fn = mean_reversion_signal

        else:
            raise HTTPException(400, f"Unknown strategy '{strat}'")

        universe_result = engine.run(long_df, signal_fn, strategy_name=strat)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Backtest failed: {e}")

    # ── Build response ───────────────────────────────────────────────────── #
    metrics_df = universe_result.universe_metrics
    top10      = universe_result.top_by_sharpe(len(actual_tickers))

    leaderboard = [
        TickerResult(
            ticker         = str(row["ticker"]),
            net_sharpe     = clean_float(row.get("net_sharpe")),
            sortino        = clean_float(row.get("sortino")),
            total_return   = clean_float(row.get("total_return")),
            max_drawdown   = clean_float(row.get("max_drawdown")),
            n_trades       = int(row.get("n_trades", 0)),
            total_cost_usd = float(row.get("total_cost_usd", 0.0)),
        )
        for row in top10.to_dicts()
    ]

    sharpes = [r.net_sharpe for r in leaderboard if r.net_sharpe is not None]
    n       = len(sharpes)

    mean_sharpe   = sum(sharpes) / n if n else None
    sorted_s      = sorted(sharpes)
    median_sharpe = sorted_s[n // 2] if n else None
    pct_pos       = sum(1 for s in sharpes if s > 0) / n if n else None
    best          = leaderboard[0].ticker  if leaderboard else None
    worst         = leaderboard[-1].ticker if leaderboard else None

    return UniverseResponse(
        tickers_run         = len(actual_tickers),
        strategy            = strat,
        leaderboard         = leaderboard,
        mean_sharpe         = clean_float(mean_sharpe),
        median_sharpe       = clean_float(median_sharpe),
        pct_positive_sharpe = clean_float(pct_pos),
        best_ticker         = best,
        worst_ticker        = worst,
    )
