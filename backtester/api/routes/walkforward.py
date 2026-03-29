"""api/routes/walkforward.py — POST /api/walkforward."""
from __future__ import annotations

import datetime as dt
import inspect
import math
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backtester.data.handler import DataHandler
from backtester.engine.walkforward import WalkForwardOptimizer
from backtester.strategy.signals import mean_reversion_signal, momentum_signal

from api.deps import clean_float, granularity_enum, make_engine, ts_to_str

router = APIRouter()


class WalkForwardRequest(BaseModel):
    ticker:          str                      = "NVO"
    strategy:        str                      = "mean_reversion"
    start_date:      str                      = "2024-01-01"
    end_date:        str                      = "2025-01-01"
    granularity:     str                      = "daily"
    train_bars:      int                      = 120
    test_bars:       int                      = 21
    step_bars:       int | None               = None
    optimise_on:     str                      = "sharpe"
    initial_capital: float                    = 100_000.0
    shares_per_unit: int                      = 1_000
    param_grid:      dict[str, list[float]]   = {
        "entry_z": [1.0, 1.5, 2.0],
        "exit_z":  [0.3, 0.5],
    }


class FoldOut(BaseModel):
    fold_index:  int
    train_start: str
    train_end:   str
    test_start:  str
    test_end:    str
    best_params: dict[str, Any]
    is_sharpe:   float | None
    oos_sharpe:  float | None
    oos_return:  float | None
    oos_mdd:     float | None


class WalkForwardResponse(BaseModel):
    folds:                list[FoldOut]
    aggregate_oos_sharpe: float | None
    is_sharpe_mean:       float | None
    stability_score:      float | None
    sharpe_degradation:   float | None
    n_folds:              int


@router.post("/walkforward", response_model=WalkForwardResponse)
def run_walkforward(req: WalkForwardRequest) -> WalkForwardResponse:
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
    if "signal" in price_df.columns:
        price_df = price_df.drop("signal")

    signal_fn = (
        momentum_signal if req.strategy == "momentum"
        else mean_reversion_signal
    )

    # Filter param_grid to only keys the signal function actually accepts
    accepted = set(inspect.signature(signal_fn).parameters)
    filtered_grid = {k: v for k, v in req.param_grid.items() if k in accepted}
    if not filtered_grid:
        # Fallback: run with fixed default params (no optimisation)
        filtered_grid = {"use_event_hedge": [False]}

    engine = make_engine(req.initial_capital, req.shares_per_unit, 0.05, req.granularity)

    optimizer = WalkForwardOptimizer(
        engine      = engine,
        signal_fn   = signal_fn,
        param_grid  = filtered_grid,
        train_bars  = req.train_bars,
        test_bars   = req.test_bars,
        step_bars   = req.step_bars,
        optimise_on = req.optimise_on,
    )

    try:
        result = optimizer.run(req.ticker, price_df, strategy_name=req.strategy)
    except ValueError as e:
        raise HTTPException(400, str(e))

    folds = [
        FoldOut(
            fold_index  = f.fold_index,
            train_start = ts_to_str(f.train_start),
            train_end   = ts_to_str(f.train_end),
            test_start  = ts_to_str(f.test_start),
            test_end    = ts_to_str(f.test_end),
            best_params = f.best_params,
            is_sharpe   = clean_float(f.is_sharpe),
            oos_sharpe  = clean_float(f.oos_sharpe),
            oos_return  = clean_float(f.oos_return),
            oos_mdd     = clean_float(f.oos_mdd),
        )
        for f in result.folds
    ]

    return WalkForwardResponse(
        folds                = folds,
        aggregate_oos_sharpe = clean_float(result.aggregate_oos_sharpe),
        is_sharpe_mean       = clean_float(result.is_sharpe_mean),
        stability_score      = clean_float(result.stability_score),
        sharpe_degradation   = clean_float(result.sharpe_degradation),
        n_folds              = result.n_folds,
    )
