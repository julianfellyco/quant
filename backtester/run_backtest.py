"""
run_backtest.py — End-to-end demo: NVO vs PFE, Momentum vs Mean Reversion.

Produces a full performance report with:
  - Sharpe Ratio (gross and net of transaction costs)
  - Sortino Ratio
  - Maximum Drawdown
  - Event-window P&L decomposition
  - Cost sensitivity analysis

Run:
    cd /Users/julianfellyco/backtester
    pip install polars          # required
    pip install yfinance        # optional — falls back to synthetic data
    PYTHONPATH=.. python run_backtest.py
"""

from __future__ import annotations

import datetime as dt

from backtester.data.handler import DataHandler, Granularity
from backtester.engine.vectorized import VectorizedEngine
from backtester.engine.report import print_report, build_comparison_table
from backtester.stats.metrics import compute_event_window_report
from backtester.strategy.signals import mean_reversion_signal, momentum_signal

TICKERS = ["PFE", "NVO"]
START   = dt.date(2024, 1,  1)
END     = dt.date(2025, 1,  1)

# Switch to Granularity.MINUTE for 1-min bar simulation
# (requires data_cache/PFE_1min.parquet and NVO_1min.parquet,
#  or will generate synthetic 1-min data automatically)
GRANULARITY = Granularity.DAILY


def main() -> None:
    # ── 1. Load data ─────────────────────────────────────────────── #
    print(f"\nLoading {GRANULARITY.name} data for {TICKERS}...")
    handler = DataHandler(TICKERS, granularity=GRANULARITY)
    handler.load(start=START, end=END)

    for t in TICKERS:
        df = handler[t]
        print(f"  {t}: {len(df):,} bars | "
              f"${float(df['close'][0]):.2f} → ${float(df['close'][-1]):.2f}")

    engine = VectorizedEngine(
        initial_capital = 100_000,
        shares_per_unit = 1_000,
        risk_free_rate  = 0.05,     # Fed funds ~5% in early 2024
        ann_factor      = handler.ann_factor,
    )

    # ── 2. Run strategies ─────────────────────────────────────────── #
    STRATEGIES = {
        "momentum":       momentum_signal,
        "mean_reversion": mean_reversion_signal,
    }
    results = []

    print("\nRunning strategies...")
    for ticker in TICKERS:
        price_df = handler[ticker]
        for name, signal_fn in STRATEGIES.items():
            signals  = signal_fn(price_df, use_event_hedge=True)
            aligned  = handler.align_signals(ticker, signals)
            result   = engine.run(ticker, aligned, name)
            results.append(result)

    # ── 3. Full report ────────────────────────────────────────────── #
    print_report(results, START, END, granularity=GRANULARITY.name)

    # ── 4. Event-window breakdown ─────────────────────────────────── #
    print("\n" + "═"*60)
    print("  EVENT-WINDOW P&L DECOMPOSITION")
    print("═"*60)
    for result in results:
        event_mask = handler[result.ticker]["is_event_window"]
        report = compute_event_window_report(
            ticker          = result.ticker,
            strategy_name   = result.strategy_name,
            equity_curve    = result.equity_curve,
            is_event_window = event_mask,
        )
        print(report.summary())

    # ── 5. Comparison table as Polars DataFrame ───────────────────── #
    tbl = build_comparison_table(results)
    print("\nComparison DataFrame (Polars):")
    print(tbl)


if __name__ == "__main__":
    main()
