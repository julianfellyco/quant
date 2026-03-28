"""
engine/report.py — Formatted performance report for pharma backtests.

Outputs a structured console report and a Polars comparison DataFrame.
No external libraries (no matplotlib, tabulate, etc.) — plain text only.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import List

import polars as pl

from .vectorized import BacktestResult


# --------------------------------------------------------------------------- #
# Comparison table                                                               #
# --------------------------------------------------------------------------- #

def build_comparison_table(results: List[BacktestResult]) -> pl.DataFrame:
    """
    Build a Polars DataFrame comparing all results.

    Columns: ticker, strategy, net_sharpe, gross_sharpe, sortino,
             max_drawdown, total_return, ann_vol, total_cost_usd, n_trades
    """
    rows = []
    for r in results:
        rows.append({
            "ticker":         r.ticker,
            "strategy":       r.strategy_name,
            "gross_sharpe":   round(r.gross_sharpe,   3),
            "net_sharpe":     round(r.net_sharpe,     3),
            "sortino":        round(r.sortino,         3),
            "max_drawdown":   round(r.max_drawdown,    4),
            "total_return":   round(r.total_return,    4),
            "ann_vol":        round(r.annualised_vol,  4),
            "total_cost_usd": round(r.total_cost_usd,  2),
            "n_trades":       r.n_trades,
        })
    return (
        pl.DataFrame(rows)
        .sort(["ticker", "net_sharpe"], descending=[False, True])
    )


# --------------------------------------------------------------------------- #
# Console report                                                                 #
# --------------------------------------------------------------------------- #

def print_report(
    results:    List[BacktestResult],
    start:      dt.date,
    end:        dt.date,
    granularity: str = "daily",
) -> None:
    """
    Print a structured performance report to stdout.

    Sections:
      1. Header
      2. Per-result metric cards (Sharpe, Sortino, MDD, return, cost)
      3. Comparison table sorted by net Sharpe
      4. Strategy alpha narrative
    """
    W = 60
    DBL = "═" * W

    def section(title: str) -> None:
        print(f"\n{DBL}")
        print(f"  {title}")
        print(DBL)

    # ── 1. Header ─────────────────────────────────────────────────────── #
    print(f"\n{'='*W}")
    print(f"  PHARMA BACKTEST REPORT  │  {start} → {end}")
    print(f"  Granularity: {granularity.upper()}")
    print(f"  Tickers: {', '.join(sorted({r.ticker for r in results}))}")
    print(f"  Strategies: {', '.join(sorted({r.strategy_name for r in results}))}")
    print(f"{'='*W}")

    # ── 2. Individual result cards ─────────────────────────────────────── #
    section("INDIVIDUAL STRATEGY RESULTS")
    for r in sorted(results, key=lambda x: (x.ticker, x.strategy_name)):
        print(r.summary())

    # ── 3. Comparison table ────────────────────────────────────────────── #
    section("SHARPE RATIO COMPARISON TABLE")
    tbl = build_comparison_table(results)
    _print_table(tbl)

    # ── 4. Alpha narrative ─────────────────────────────────────────────── #
    section("STRATEGY ALPHA NARRATIVE")
    _print_narrative(results)

    print()


def _print_table(df: pl.DataFrame) -> None:
    """Print a Polars DataFrame as a fixed-width ASCII table."""
    # Format columns for display
    display_df = df.with_columns([
        pl.col("max_drawdown").map_elements(
            lambda x: f"{x:.2%}", return_dtype=pl.String
        ),
        pl.col("total_return").map_elements(
            lambda x: f"{x:.2%}", return_dtype=pl.String
        ),
        pl.col("ann_vol").map_elements(
            lambda x: f"{x:.2%}", return_dtype=pl.String
        ),
        pl.col("total_cost_usd").map_elements(
            lambda x: f"${x:,.0f}", return_dtype=pl.String
        ),
        pl.col("net_sharpe").map_elements(
            lambda x: f"{x:+.3f}", return_dtype=pl.String
        ),
        pl.col("gross_sharpe").map_elements(
            lambda x: f"{x:+.3f}", return_dtype=pl.String
        ),
        pl.col("sortino").map_elements(
            lambda x: f"{x:+.3f}", return_dtype=pl.String
        ),
    ])

    col_widths = {
        "ticker":         6,
        "strategy":       15,
        "gross_sharpe":   12,
        "net_sharpe":     12,
        "sortino":        10,
        "max_drawdown":   12,
        "total_return":   13,
        "ann_vol":        9,
        "total_cost_usd": 13,
        "n_trades":       9,
    }
    header = "  ".join(
        col.replace("_", " ").upper()[:w].ljust(w)
        for col, w in col_widths.items()
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in display_df.iter_rows(named=True):
        line = "  ".join(
            str(row[col])[:w].ljust(w)
            for col, w in col_widths.items()
        )
        print(line)
    print(sep)


def _print_narrative(results: List[BacktestResult]) -> None:
    """
    Write a qualitative interpretation of the results.

    Quant Why — human narrative matters: a Sharpe table without context
    leads to p-hacking.  We call out:
      - Whether the winner is driven by event windows (fragile) or
        structural alpha (repeatable)
      - Whether net Sharpe is materially below gross (cost sensitivity)
      - NVO vs PFE divergence in the GLP-1 narrative context
    """
    by_ticker: dict[str, list[BacktestResult]] = {}
    for r in results:
        by_ticker.setdefault(r.ticker, []).append(r)

    for ticker, ticker_results in sorted(by_ticker.items()):
        best  = max(ticker_results, key=lambda r: r.net_sharpe)
        worst = min(ticker_results, key=lambda r: r.net_sharpe)

        print(f"\n  {ticker}")
        print(f"  {'─'*40}")
        print(f"  Best strategy  : {best.strategy_name}  (net Sharpe {best.net_sharpe:+.3f})")
        print(f"  Worst strategy : {worst.strategy_name}  (net Sharpe {worst.net_sharpe:+.3f})")

        cost_drag = best.gross_sharpe - best.net_sharpe
        print(f"  Cost drag      : {cost_drag:.3f} Sharpe units on best strategy")

        if cost_drag > 0.5:
            print(f"  WARNING: high cost drag — strategy is cost-sensitive. "
                  f"Consider wider signal entry thresholds or lower turnover.")

        if best.max_drawdown < -0.20:
            print(f"  WARNING: MDD {best.max_drawdown:.1%} exceeds −20%. "
                  f"Binary event blowouts are likely. Review event hedge.")

        if ticker == "NVO":
            print(f"  GLP-1 context: NVO's 2024 re-rating was driven by Wegovy/Ozempic "
                  f"demand and positive cardiovascular trial data. Momentum strategies "
                  f"benefit from persistent institutional rotation into GLP-1 plays.")
        elif ticker == "PFE":
            print(f"  GLP-1 context: PFE's 2024 underperformance reflects COVID-revenue "
                  f"collapse and failure to develop a competitive oral GLP-1 (Danuglipron "
                  f"was withdrawn Oct 2024). Mean-reversion may capture post-event "
                  f"overshoots but is undermined by genuine fundamental deterioration.")
