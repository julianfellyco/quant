"""api/routes/backtest.py — POST /api/backtest."""
from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backtester.data.handler import DataHandler
from backtester.engine.vectorized import BacktestResult
from backtester.stats.metrics import compute_event_window_report, compare_sharpe
from backtester.strategy.signals import mean_reversion_signal, momentum_signal

from api.deps import (
    clean_float, granularity_enum, make_engine,
    serialize_equity_curve, serialize_window_stats,
)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Request / Response models                                                      #
# --------------------------------------------------------------------------- #

class BacktestRequest(BaseModel):
    tickers:         list[str]  = ["NVO", "PFE"]
    start_date:      str        = "2024-01-01"
    end_date:        str        = "2025-01-01"
    strategies:      list[str]  = ["momentum", "mean_reversion"]
    granularity:     str        = "daily"
    initial_capital: float      = 100_000.0
    shares_per_unit: int        = 1_000
    risk_free_rate:  float      = 0.05
    use_event_hedge: bool       = True
    entry_z:         float      = 1.5
    exit_z:          float      = 0.3
    position_sizer_type: str | None = None   # "fixed_fractional" | "volatility_target" | "kelly"
    target_vol: float = 0.15                  # used by volatility_target sizer
    stop_loss_type: str | None = None         # "atr" | "fixed_pct"
    stop_loss_param: float = 0.05             # stop_pct (fixed) or multiplier (atr)


class MetricsSummary(BaseModel):
    ticker:         str
    strategy:       str
    gross_sharpe:   float | None
    net_sharpe:     float | None
    sortino:        float | None
    max_drawdown:   float | None
    total_return:   float | None
    annualised_vol: float | None
    total_cost_usd: float
    n_trades:       int


class SingleBacktestResult(BaseModel):
    metrics:      MetricsSummary
    equity_curve: list[dict]
    event_decomp: dict[str, dict]


class BacktestResponse(BaseModel):
    results:          list[SingleBacktestResult]
    comparison_table: list[dict]


# --------------------------------------------------------------------------- #
# Route                                                                          #
# --------------------------------------------------------------------------- #

@router.post("/backtest", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest) -> BacktestResponse:
    try:
        start = dt.date.fromisoformat(req.start_date)
        end   = dt.date.fromisoformat(req.end_date)
    except ValueError as e:
        raise HTTPException(400, f"Invalid date: {e}")

    try:
        gran = granularity_enum(req.granularity)
    except ValueError as e:
        raise HTTPException(400, str(e))

    valid_strategies = {"momentum", "mean_reversion"}
    bad = set(req.strategies) - valid_strategies
    if bad:
        raise HTTPException(400, f"Unknown strategies: {bad}")

    try:
        handler = DataHandler(req.tickers, granularity=gran)
        handler.load(start=start, end=end)
    except Exception as e:
        raise HTTPException(500, f"Data loading failed: {e}")

    engine = make_engine(
        req.initial_capital, req.shares_per_unit,
        req.risk_free_rate, req.granularity,
    )

    def _signal(name: str, df):
        if name == "momentum":
            return momentum_signal(df, use_event_hedge=req.use_event_hedge)
        return mean_reversion_signal(
            df,
            entry_z         = req.entry_z,
            exit_z          = req.exit_z,
            use_event_hedge = req.use_event_hedge,
        )

    results:    list[SingleBacktestResult] = []
    bt_results: list[BacktestResult]       = []

    for ticker in req.tickers:
        price_df = handler[ticker]
        for strat in req.strategies:
            sig     = _signal(strat, price_df)
            aligned = handler.align_signals(ticker, sig)
            result  = engine.run(ticker, aligned, strat)
            bt_results.append(result)

            event_mask = handler[ticker]["is_event_window"]
            ew         = compute_event_window_report(
                ticker, strat, result.equity_curve, event_mask,
                risk_free_rate=req.risk_free_rate,
            )

            results.append(SingleBacktestResult(
                metrics = MetricsSummary(
                    ticker         = result.ticker,
                    strategy       = result.strategy_name,
                    gross_sharpe   = clean_float(result.gross_sharpe),
                    net_sharpe     = clean_float(result.net_sharpe),
                    sortino        = clean_float(result.sortino),
                    max_drawdown   = clean_float(result.max_drawdown),
                    total_return   = clean_float(result.total_return),
                    annualised_vol = clean_float(result.annualised_vol),
                    total_cost_usd = result.total_cost_usd,
                    n_trades       = result.n_trades,
                ),
                equity_curve = serialize_equity_curve(result),
                event_decomp = {
                    "full":      serialize_window_stats(ew.full_period),
                    "event":     serialize_window_stats(ew.event_windows),
                    "non_event": serialize_window_stats(ew.non_event),
                },
            ))

    comparison = compare_sharpe(bt_results).to_dicts()
    # Clean non-finite floats from comparison rows
    for row in comparison:
        for k, v in row.items():
            row[k] = clean_float(v) if isinstance(v, float) else v

    return BacktestResponse(results=results, comparison_table=comparison)
