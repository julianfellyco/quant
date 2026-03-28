"""api/routes/stress.py — POST /api/stress."""
from __future__ import annotations

import datetime as dt
import math

import polars as pl
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backtester.data.events import get_event_dates
from backtester.data.handler import DataHandler
from backtester.engine.stress import EventShuffler
from backtester.strategy.signals import mean_reversion_signal, momentum_signal

from api.deps import clean_float, granularity_enum, make_engine

router = APIRouter()


class StressRequest(BaseModel):
    ticker:          str   = "NVO"
    strategy:        str   = "momentum"
    start_date:      str   = "2024-01-01"
    end_date:        str   = "2025-01-01"
    granularity:     str   = "daily"
    n_simulations:   int   = 200
    max_shift_days:  int   = 5
    seed:            int   = 42
    initial_capital: float = 100_000.0
    shares_per_unit: int   = 1_000
    entry_z:         float = 1.5
    exit_z:          float = 0.3
    use_event_hedge: bool  = True


class HistogramBin(BaseModel):
    bin_start: float
    bin_end:   float
    count:     int


class StressResponse(BaseModel):
    base_sharpe:         float | None
    base_return:         float | None
    base_mdd:            float | None
    p5_sharpe:           float | None
    p95_sharpe:          float | None
    fragility_score:     float
    worst_sharpe:        float | None
    best_sharpe:         float | None
    n_simulations:       int
    max_shift_days:      int
    histogram_bins:      list[HistogramBin]
    sharpe_distribution: list[float | None]


def _make_histogram(values: list[float], n_bins: int = 20) -> list[HistogramBin]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return []
    lo, hi = min(finite), max(finite)
    if lo == hi:
        hi = lo + 1e-9
    width  = (hi - lo) / n_bins
    counts = [0] * n_bins
    for v in finite:
        idx = min(int((v - lo) / width), n_bins - 1)
        counts[idx] += 1
    return [
        HistogramBin(
            bin_start = lo + i * width,
            bin_end   = lo + (i + 1) * width,
            count     = counts[i],
        )
        for i in range(n_bins)
    ]


@router.post("/stress", response_model=StressResponse)
def run_stress(req: StressRequest) -> StressResponse:
    try:
        start = dt.date.fromisoformat(req.start_date)
        end   = dt.date.fromisoformat(req.end_date)
    except ValueError as e:
        raise HTTPException(400, f"Invalid date: {e}")

    try:
        gran = granularity_enum(req.granularity)
    except ValueError as e:
        raise HTTPException(400, str(e))

    handler = DataHandler([req.ticker], granularity=gran)
    handler.load(start=start, end=end)

    price_df = handler[req.ticker]
    if req.strategy == "momentum":
        raw_sig = momentum_signal(price_df, use_event_hedge=req.use_event_hedge)
    else:
        raw_sig = mean_reversion_signal(
            price_df, entry_z=req.entry_z, exit_z=req.exit_z,
            use_event_hedge=req.use_event_hedge,
        )

    # Pre-shift signal for PiT compliance
    signal = raw_sig.shift(1).fill_null(0.0)

    events  = get_event_dates(req.ticker, start, end)
    engine  = make_engine(req.initial_capital, req.shares_per_unit, 0.05, req.granularity)
    shuffler = EventShuffler(engine, n_simulations=req.n_simulations, seed=req.seed)

    base_cols = {"signal", "is_event_window"}
    aligned_df = price_df.drop([c for c in base_cols if c in price_df.columns])

    result = shuffler.run(
        ticker        = req.ticker,
        aligned_df    = aligned_df,
        signal_series = signal,
        events        = events,
        max_shift_days = req.max_shift_days,
        strategy_name  = req.strategy,
    )

    # Replace NaN in distribution with None for JSON
    clean_dist = [
        (v if math.isfinite(v) else None)
        for v in result.sharpe_distribution
    ]

    return StressResponse(
        base_sharpe         = clean_float(result.base_sharpe),
        base_return         = clean_float(result.base_return),
        base_mdd            = clean_float(result.base_mdd),
        p5_sharpe           = clean_float(result.p5_sharpe),
        p95_sharpe          = clean_float(result.p95_sharpe),
        fragility_score     = result.fragility_score,
        worst_sharpe        = clean_float(result.worst_sharpe),
        best_sharpe         = clean_float(result.best_sharpe),
        n_simulations       = result.n_simulations,
        max_shift_days      = result.max_shift_days,
        histogram_bins      = _make_histogram(result.sharpe_distribution),
        sharpe_distribution = clean_dist,
    )
