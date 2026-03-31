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
    benchmark_ticker: str = "SPY"


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


class BenchmarkStats(BaseModel):
    ticker:       str
    total_return: float | None
    sharpe:       float | None


class AlphaDecomposition(BaseModel):
    alpha_annual:      float | None
    beta:              float | None
    information_ratio: float | None
    tracking_error:    float | None
    up_capture:        float | None
    down_capture:      float | None


class BacktestResponse(BaseModel):
    results:           list[SingleBacktestResult]
    comparison_table:  list[dict]
    benchmark:         BenchmarkStats | None = None
    alpha_decomposition: AlphaDecomposition | None = None


# --------------------------------------------------------------------------- #
# Route                                                                          #
# --------------------------------------------------------------------------- #

@router.post("/backtest", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest) -> BacktestResponse:
    """Run a vectorized backtest for one or more tickers and strategies.

    Fetches price data via DataHandler (parquet cache + yfinance fallback),
    computes signals, runs VectorizedEngine for each ticker/strategy combination,
    and returns per-run metrics, equity curves, event-window decomposition, and
    optional benchmark/alpha statistics against the requested benchmark ticker.

    Args:
        req: BacktestRequest containing tickers, date range, strategy names,
             granularity, capital parameters, position-sizer config, stop-loss
             config, and benchmark ticker.

    Returns:
        BacktestResponse with per-run results, a cross-run comparison table,
        benchmark statistics, and alpha decomposition (all best-effort; benchmark
        enrichment is skipped silently if data is unavailable).

    Raises:
        HTTPException 400: Invalid date format, unsupported granularity, or
            unknown strategy name.
        HTTPException 500: Data loading failure.
    """
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

    # ── Benchmark & alpha decomposition ─────────────────────────────────── #
    bench_stats_resp = None
    alpha_decomp_resp = None
    try:
        from backtester.data.benchmark import get_benchmark_stats, get_benchmark_returns
        from backtester.stats.metrics import compute_benchmark_stats

        bench_stats = get_benchmark_stats(req.start_date, req.end_date, req.benchmark_ticker)
        bench_stats_resp = BenchmarkStats(
            ticker=bench_stats["ticker"],
            total_return=bench_stats.get("total_return"),
            sharpe=bench_stats.get("sharpe"),
        )

        # Use net returns from first result for alpha decomp
        if bt_results:
            bench_rets = get_benchmark_returns(req.start_date, req.end_date, req.benchmark_ticker)
            if not bench_rets.is_empty():
                strat_rets = bt_results[0].equity_curve["net_log_ret"]
                alpha_dict = compute_benchmark_stats(strat_rets, bench_rets)
                alpha_decomp_resp = AlphaDecomposition(
                    alpha_annual=      clean_float(alpha_dict.get("alpha_annual")),
                    beta=              clean_float(alpha_dict.get("beta")),
                    information_ratio= clean_float(alpha_dict.get("information_ratio")),
                    tracking_error=    clean_float(alpha_dict.get("tracking_error")),
                    up_capture=        clean_float(alpha_dict.get("up_capture")),
                    down_capture=      clean_float(alpha_dict.get("down_capture")),
                )
    except Exception:
        pass  # benchmark enrichment is best-effort

    return BacktestResponse(
        results=results,
        comparison_table=comparison,
        benchmark=bench_stats_resp,
        alpha_decomposition=alpha_decomp_resp,
    )
