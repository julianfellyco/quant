"""
stats/metrics.py — Performance analytics with pharma-specific event windows.

Quant Why — Why separate event-window statistics?
A standard backtest on NVO during 2024 would show outstanding Sharpe Ratio,
but a large fraction of P&L is attributable to a single event: the February
2024 cardiovascular outcomes data and the June 2024 FDA approval of the
new cardiovascular indication. If your strategy was simply long NVO on those
dates, the Sharpe looks great — but it's essentially a binary bet, not
alpha.

By reporting event-window statistics separately, we distinguish:
  1. *Structural alpha* — P&L earned on non-event days (repeatable)
  2. *Event alpha*      — P&L concentrated in binary event windows (fragile)

A strategy with a 1.5 Sharpe but 80% of P&L from event windows is very
different (and far riskier) than a 1.0 Sharpe with evenly distributed P&L.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Optional

import polars as pl


TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Window-aware statistics                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class WindowStats:
    """Performance breakdown over a specific date window."""
    label:          str
    n_days:         int
    total_log_ret:  float
    ann_sharpe:     float
    max_drawdown:   float
    pct_of_total:   float    # this window's P&L as fraction of full-period P&L


@dataclass
class EventWindowReport:
    ticker:          str
    strategy_name:   str
    full_period:     WindowStats
    event_windows:   WindowStats
    non_event:       WindowStats

    def summary(self) -> str:
        def fmt(ws: WindowStats) -> str:
            return (
                f"  {ws.label:<25} "
                f"days={ws.n_days:<5} "
                f"ret={ws.total_log_ret:+.2%}  "
                f"sharpe={ws.ann_sharpe:+.2f}  "
                f"mdd={ws.max_drawdown:.2%}  "
                f"pct_total={ws.pct_of_total:.1%}"
            )
        header = f"\n{'═'*80}\n  Event-Window Report │ {self.ticker} │ {self.strategy_name}\n{'═'*80}"
        return "\n".join([
            header,
            fmt(self.full_period),
            fmt(self.event_windows),
            fmt(self.non_event),
            "═" * 80,
        ])


def compute_event_window_report(
    ticker:           str,
    strategy_name:    str,
    equity_curve:     pl.DataFrame,   # from BacktestResult.equity_curve
    is_event_window:  pl.Series,      # boolean mask aligned to equity_curve
    risk_free_rate:   float = 0.05,
) -> EventWindowReport:
    """
    Compute full-period, event-window, and non-event performance metrics.

    Args:
        equity_curve:    DataFrame with columns [date, equity, net_log_ret]
        is_event_window: boolean Series, True on event window days
    """
    rfr_daily = math.log(1 + risk_free_rate) / TRADING_DAYS_PER_YEAR

    df = equity_curve.with_columns(is_event_window.alias("is_event"))

    full_ret  = df["net_log_ret"]
    event_ret = df.filter(pl.col("is_event"))["net_log_ret"]
    non_ret   = df.filter(~pl.col("is_event"))["net_log_ret"]

    total_pnl = float(full_ret.sum())

    full_stats  = _window_stats("Full period",        full_ret,  rfr_daily, total_pnl)
    event_stats = _window_stats("Event windows",      event_ret, rfr_daily, total_pnl)
    non_stats   = _window_stats("Non-event days",     non_ret,   rfr_daily, total_pnl)

    return EventWindowReport(
        ticker        = ticker,
        strategy_name = strategy_name,
        full_period   = full_stats,
        event_windows = event_stats,
        non_event     = non_stats,
    )


def _window_stats(
    label:     str,
    log_rets:  pl.Series,
    rfr_daily: float,
    total_pnl: float,
) -> WindowStats:
    n = len(log_rets)
    if n == 0:
        return WindowStats(label, 0, 0.0, float("nan"), 0.0, 0.0)

    total_ret = float(log_rets.sum())
    excess    = log_rets - rfr_daily
    mu        = float(excess.mean() or 0.0)
    sigma     = float(excess.std()  or 1e-9)
    sharpe    = mu / sigma * math.sqrt(TRADING_DAYS_PER_YEAR) if sigma > 0 else float("nan")
    mdd       = _running_mdd(log_rets)
    pct       = total_ret / total_pnl if total_pnl != 0 else float("nan")

    return WindowStats(
        label         = label,
        n_days        = n,
        total_log_ret = total_ret,
        ann_sharpe    = sharpe,
        max_drawdown  = mdd,
        pct_of_total  = pct,
    )


def _running_mdd(log_rets: pl.Series) -> float:
    """MDD from a series of log returns (reconstructs equity curve locally)."""
    if len(log_rets) == 0:
        return 0.0
    equity      = log_rets.cum_sum().exp()
    running_max = equity.cum_max()
    drawdowns   = equity / running_max - 1.0
    return float(drawdowns.min() or 0.0)


# --------------------------------------------------------------------------- #
# Comparative Sharpe table                                                      #
# --------------------------------------------------------------------------- #

def compare_sharpe(
    results: list,   # list[BacktestResult]
) -> pl.DataFrame:
    """
    Build a comparison DataFrame of Sharpe ratios across tickers and strategies.

    Returns:
        pl.DataFrame with columns:
            ticker, strategy, gross_sharpe, net_sharpe, sortino,
            max_drawdown, total_return, total_cost_usd, n_trades
    """
    rows = [
        {
            "ticker":         r.ticker,
            "strategy":       r.strategy_name,
            "gross_sharpe":   round(r.gross_sharpe,  3),
            "net_sharpe":     round(r.net_sharpe,    3),
            "sortino":        round(r.sortino,        3),
            "max_drawdown":   round(r.max_drawdown,   4),
            "total_return":   round(r.total_return,   4),
            "ann_vol":        round(r.annualised_vol, 4),
            "total_cost_usd": round(r.total_cost_usd, 2),
            "n_trades":       r.n_trades,
        }
        for r in results
    ]
    return pl.DataFrame(rows).sort(["ticker", "net_sharpe"], descending=[False, True])
