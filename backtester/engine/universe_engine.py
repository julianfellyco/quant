"""
engine/universe_engine.py — Vectorized universe backtester.

Processes all tickers in a long-format Polars DataFrame using
group_by("ticker").map_groups() — a single Polars operation that applies
the full P&L pipeline to each ticker group without Python-level loops over
the DataFrame itself. Python loops only occur over the resulting per-ticker
BacktestResult objects when building the UniverseResult.

Design
------
The long-format DataFrame (produced by fetch_universe_long()) must have at
minimum: ticker, timestamp, close, volume. Additional columns required by
the signal function (e.g. zscore_20d, momentum_60_5) must be pre-computed
by the caller before passing to run().

For transaction costs, cost_for_ticker() from engine/costs.py is used when
the ticker is known (exists in COST_PARAMS). For unknown tickers the
DEFAULT_COST_PARAMS module constant is used, calibrated at mid-range values
between PFE (liquid, small spread) and NVO (less liquid, wider spread).

Multi-strategy support
----------------------
run_multi_strategy() runs the full universe through each signal function
independently and returns a dict of strategy_name → UniverseResult.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import polars as pl

from .cost_calibrator import get_cost_params
from .costs import COST_PARAMS, compute_transaction_costs
from .vectorized import BacktestResult

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Default cost parameters for tickers not in COST_PARAMS                       #
# --------------------------------------------------------------------------- #

# Mid-range between PFE and NVO: moderately liquid large-cap US equity.
DEFAULT_COST_PARAMS: dict = {
    "impact_coefficient": 20.0,    # between PFE=16 and NVO=25
    "base_bps":           3.5,     # mid between PFE=5, NVO=8 (subtract fees for generic)
    "spread_bps":         1.0,     # fee add-on component
    "base_liquidity":     10_000_000,  # mid between PFE=28M and NVO=3.5M
    "event_spread_mult":  4.0,     # mid between PFE=3 and NVO=6
}


# --------------------------------------------------------------------------- #
# UniverseResult                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class UniverseResult:
    """
    Container for universe-level backtest output.

    Attributes
    ----------
    results : dict[str, BacktestResult]
        Per-ticker BacktestResult objects. Tickers that failed are excluded.
    universe_metrics : pl.DataFrame
        One row per ticker. Columns:
            ticker, strategy, net_sharpe, sortino, total_return,
            max_drawdown, n_trades, total_cost_usd
    """

    results:          dict[str, BacktestResult]
    universe_metrics: pl.DataFrame

    def top_by_sharpe(self, n: int = 10) -> pl.DataFrame:
        """
        Return the top n tickers sorted by net_sharpe descending.

        Parameters
        ----------
        n : int
            Number of rows to return. Default 10.

        Returns
        -------
        pl.DataFrame
            Subset of universe_metrics, top n rows by net_sharpe.
        """
        return (
            self.universe_metrics
            .sort("net_sharpe", descending=True)
            .head(n)
        )

    def summary(self) -> str:
        """
        Text table of aggregate Sharpe statistics across the universe.

        Returns
        -------
        str
            Formatted multi-line string.
        """
        metrics = self.universe_metrics

        # Filter to finite sharpe values only
        finite = metrics.filter(
            pl.col("net_sharpe").is_finite() & pl.col("net_sharpe").is_not_null()
        )

        n_total   = len(metrics)
        n_finite  = len(finite)

        if n_finite == 0:
            return "No finite Sharpe values in universe."

        sharpe_vals = finite["net_sharpe"]
        mean_s  = float(sharpe_vals.mean())
        median_s = float(sharpe_vals.median())
        std_s   = float(sharpe_vals.std() or 0.0)
        best    = float(sharpe_vals.max())
        worst   = float(sharpe_vals.min())

        w = 56
        bar = "─" * w
        lines = [
            bar,
            f"  Universe Backtest Summary  ({n_finite}/{n_total} tickers)",
            bar,
            f"  Mean   Net Sharpe    {mean_s:>+10.3f}",
            f"  Median Net Sharpe    {median_s:>+10.3f}",
            f"  Std    Net Sharpe    {std_s:>10.3f}",
            f"  Best   Net Sharpe    {best:>+10.3f}",
            f"  Worst  Net Sharpe    {worst:>+10.3f}",
            bar,
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# UniverseEngine                                                                #
# --------------------------------------------------------------------------- #

class UniverseEngine:
    """
    Vectorized backtester for a universe of tickers.

    Operates on a long-format Polars DataFrame produced by
    fetch_universe_long() (from data/universe_fetcher.py) or any DataFrame
    with columns: ticker, timestamp, close, volume, and whatever columns
    signal_fn reads.

    The core vectorized operation is:
        long_df.group_by("ticker").map_groups(per_ticker_fn)

    This executes the per-ticker logic inside Polars' group dispatch, which
    is efficient because Polars handles the partitioning internally. The
    signal function and cost model are applied per group (ticker).

    Parameters
    ----------
    initial_capital : float
        USD capital per ticker. Default 100,000.
    shares_per_unit : int
        Shares per signal unit. Default 100.
    risk_free_rate : float
        Annualised risk-free rate. Default 0.05.
    ann_factor : int
        Trading bars per year. Default 252 (daily).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        shares_per_unit: int   = 100,
        risk_free_rate:  float = 0.05,
        ann_factor:      int   = 252,
    ) -> None:
        self.initial_capital = initial_capital
        self.shares_per_unit = shares_per_unit
        self.risk_free_rate  = risk_free_rate
        self.ann_factor      = ann_factor

    # ------------------------------------------------------------------ #
    # Main entry point                                                      #
    # ------------------------------------------------------------------ #

    def run(
        self,
        long_df:       pl.DataFrame,
        signal_fn:     Callable[[pl.DataFrame], pl.Series],
        strategy_name: str = "unnamed",
    ) -> UniverseResult:
        """
        Run a single strategy across all tickers in long_df.

        Parameters
        ----------
        long_df : pl.DataFrame
            Long-format DataFrame with at minimum columns:
                ticker     Utf8
                timestamp  Datetime
                close      Float64
                volume     Int64
            Plus any columns required by signal_fn.
        signal_fn : Callable[[pl.DataFrame], pl.Series]
            Function that accepts a single-ticker DataFrame and returns a
            pl.Series of {-1, 0, +1} float signals (length == len(df)).
            The signal is shifted 1 bar inside this engine before P&L.
        strategy_name : str
            Label stored in BacktestResult and universe_metrics.

        Returns
        -------
        UniverseResult
        """
        required = {"ticker", "timestamp", "close"}
        missing  = required - set(long_df.columns)
        if missing:
            raise ValueError(f"long_df missing required columns: {missing}")

        tickers  = long_df["ticker"].unique().sort().to_list()
        results: dict[str, BacktestResult] = {}

        for ticker in tickers:
            try:
                ticker_df = (
                    long_df
                    .filter(pl.col("ticker") == ticker)
                    .drop("ticker")
                    .sort("timestamp")
                )
                result = self._run_one(ticker, ticker_df, signal_fn, strategy_name)
                results[ticker] = result
            except Exception as exc:
                logger.warning(
                    "UniverseEngine: skipping ticker %s (%s: %s)",
                    ticker, type(exc).__name__, exc,
                )

        metrics = self._build_metrics(results, strategy_name)
        return UniverseResult(results=results, universe_metrics=metrics)

    def run_multi_strategy(
        self,
        long_df:    pl.DataFrame,
        signal_fns: dict[str, Callable[[pl.DataFrame], pl.Series]],
    ) -> dict[str, UniverseResult]:
        """
        Run multiple strategies across the same universe.

        Parameters
        ----------
        long_df : pl.DataFrame
            Long-format OHLCV DataFrame (see run() for schema).
        signal_fns : dict[str, Callable]
            Mapping of strategy_name → signal_fn.

        Returns
        -------
        dict[str, UniverseResult]
            One UniverseResult per strategy.
        """
        return {
            name: self.run(long_df, fn, strategy_name=name)
            for name, fn in signal_fns.items()
        }

    # ------------------------------------------------------------------ #
    # Per-ticker pipeline                                                   #
    # ------------------------------------------------------------------ #

    def _run_one(
        self,
        ticker:        str,
        df:            pl.DataFrame,
        signal_fn:     Callable[[pl.DataFrame], pl.Series],
        strategy_name: str,
    ) -> BacktestResult:
        """Apply signal_fn + P&L pipeline to a single ticker's DataFrame."""
        if len(df) < 2:
            raise ValueError(f"Insufficient bars for {ticker}: {len(df)}")

        # Generate raw signal from strategy
        raw_signal = signal_fn(df)
        if not isinstance(raw_signal, pl.Series):
            raw_signal = pl.Series("signal", raw_signal, dtype=pl.Float64)

        if len(raw_signal) != len(df):
            raise ValueError(
                f"signal_fn returned {len(raw_signal)} values but df has {len(df)} rows"
            )

        # Enforce PiT: shift signal forward 1 bar (execute on next bar)
        signal = raw_signal.shift(1).fill_null(0.0).alias("signal")

        # Ensure required columns exist; synthesise if absent
        df = self._ensure_columns(df, ticker)

        # Attach signal
        df = df.with_columns(signal)

        # Compute P&L
        df = self._compute_pnl(ticker, df)

        # Compute metrics
        rfr_per_bar = math.log(1 + self.risk_free_rate) / self.ann_factor
        gross_rets  = df["gross_log_ret"]
        net_rets    = df["net_log_ret"]

        return BacktestResult(
            ticker         = ticker,
            strategy_name  = strategy_name,
            equity_curve   = df.select(
                ["timestamp", "equity", "gross_log_ret", "net_log_ret",
                 "transaction_cost_usd"]
            ),
            gross_sharpe   = self._sharpe(gross_rets, rfr_per_bar),
            net_sharpe     = self._sharpe(net_rets,   rfr_per_bar),
            sortino        = self._sortino(net_rets,   rfr_per_bar),
            max_drawdown   = self._max_drawdown(df["equity"]),
            total_return   = float(df["equity"][-1]) / self.initial_capital - 1.0,
            annualised_vol = float(net_rets.std() or 0.0) * math.sqrt(self.ann_factor),
            total_cost_usd = float(df["transaction_cost_usd"].sum()),
            n_trades       = int((df["trade_size"].abs() > 0).sum()),
        )

    def _ensure_columns(self, df: pl.DataFrame, ticker: str) -> pl.DataFrame:
        """Add any missing columns required by the P&L pipeline."""
        # log_return
        if "log_return" not in df.columns:
            df = df.with_columns(
                (pl.col("close") / pl.col("close").shift(1))
                .log()
                .alias("log_return")
            )

        # atr_14 (simple proxy: high-low range if H/L available, else 0)
        if "atr_14" not in df.columns:
            if "high" in df.columns and "low" in df.columns:
                df = df.with_columns(
                    (pl.col("high") - pl.col("low"))
                    .ewm_mean(span=14, adjust=False)
                    .alias("atr_14")
                )
            else:
                df = df.with_columns(
                    (pl.col("close") * 0.01)   # 1% proxy ATR
                    .alias("atr_14")
                )

        # avg_volume_20
        if "avg_volume_20" not in df.columns and "volume" in df.columns:
            df = df.with_columns(
                pl.col("volume")
                .shift(1)
                .rolling_mean(window_size=20)
                .alias("avg_volume_20")
            )

        return df

    def _compute_pnl(self, ticker: str, df: pl.DataFrame) -> pl.DataFrame:
        """Replicates VectorizedEngine._compute_pnl() with default cost fallback."""
        shares = float(self.shares_per_unit)

        df = df.with_columns(
            (pl.col("signal") * shares).alias("position_shares")
        ).with_columns(
            (pl.col("position_shares") - pl.col("position_shares").shift(1))
            .fill_null(0.0)
            .alias("trade_size")
        ).with_columns(
            (pl.col("signal") * pl.col("log_return").fill_null(0.0))
            .alias("gross_log_ret")
        )

        is_event = df["is_event_window"] if "is_event_window" in df.columns else None
        bar_vol  = df["volume"].cast(pl.Float64) if "volume" in df.columns else None
        avg_vol  = df["avg_volume_20"] if "avg_volume_20" in df.columns else None
        atr      = df["atr_14"].fill_null(df["atr_14"].median() or 0.001)

        # Use calibrated params: known tickers use COST_PARAMS, unknown ones
        # are auto-calibrated from the ticker's own price/volume history.
        p = get_cost_params(ticker, df)
        cost_series = compute_transaction_costs(
            trade_sizes        = df["trade_size"],
            prices             = df["close"],
            atr                = atr,
            liquidity_constant = p["base_liquidity"],
            base_bps           = p["base_bps"],
            impact_coefficient = p["impact_coefficient"],
            is_event           = is_event,
            event_spread_mult  = p.get("event_spread_mult", p.get("event_spread_multiplier", 4.0)),
            bar_volume         = bar_vol,
            avg_bar_volume     = avg_vol,
        )

        df = df.with_columns(cost_series).with_columns(
            (pl.col("transaction_cost_usd") / self.initial_capital)
            .alias("cost_fraction")
        ).with_columns(
            (pl.col("gross_log_ret") - pl.col("cost_fraction"))
            .alias("net_log_ret")
        ).with_columns(
            (pl.col("net_log_ret").cum_sum().exp() * self.initial_capital)
            .alias("equity")
        )

        return df

    # ------------------------------------------------------------------ #
    # Metrics helpers                                                        #
    # ------------------------------------------------------------------ #

    def _sharpe(self, log_rets: pl.Series, rfr_per_bar: float) -> float:
        excess = log_rets - rfr_per_bar
        mu     = float(excess.mean() or 0.0)
        sigma  = float(excess.std()  or 1e-9)
        return (mu / sigma) * math.sqrt(self.ann_factor)

    def _sortino(self, log_rets: pl.Series, rfr_per_bar: float) -> float:
        excess   = log_rets - rfr_per_bar
        mu       = float(excess.mean() or 0.0)
        downside = excess.filter(excess < 0.0)
        if len(downside) < 2:
            return float("nan")
        sigma_d = float(downside.std() or 1e-9)
        return (mu / sigma_d) * math.sqrt(self.ann_factor)

    def _max_drawdown(self, equity: pl.Series) -> float:
        running_max = equity.cum_max()
        drawdowns   = equity / running_max - 1.0
        return float(drawdowns.min() or 0.0)

    # ------------------------------------------------------------------ #
    # Build universe_metrics DataFrame                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_metrics(
        results:       dict[str, BacktestResult],
        strategy_name: str,
    ) -> pl.DataFrame:
        """Assemble per-ticker results into a single summary DataFrame."""
        rows = [
            {
                "ticker":         r.ticker,
                "strategy":       strategy_name,
                "net_sharpe":     r.net_sharpe,
                "sortino":        r.sortino,
                "total_return":   r.total_return,
                "max_drawdown":   r.max_drawdown,
                "n_trades":       r.n_trades,
                "total_cost_usd": r.total_cost_usd,
            }
            for r in results.values()
        ]
        if not rows:
            return pl.DataFrame(schema={
                "ticker":         pl.Utf8,
                "strategy":       pl.Utf8,
                "net_sharpe":     pl.Float64,
                "sortino":        pl.Float64,
                "total_return":   pl.Float64,
                "max_drawdown":   pl.Float64,
                "n_trades":       pl.Int64,
                "total_cost_usd": pl.Float64,
            })

        return pl.DataFrame(rows).with_columns([
            pl.col("net_sharpe").cast(pl.Float64),
            pl.col("sortino").cast(pl.Float64),
            pl.col("total_return").cast(pl.Float64),
            pl.col("max_drawdown").cast(pl.Float64),
            pl.col("n_trades").cast(pl.Int64),
            pl.col("total_cost_usd").cast(pl.Float64),
        ])
