"""
engine/stress.py — Monte Carlo event-date shuffling stress test.

Why event shuffling matters
----------------------------
A standard backtest is a single draw from the space of possible event timings.
In pharma backtesting, this is especially dangerous because:

  - P&L can be highly concentrated in 3–5 binary event windows per year.
  - If the strategy happened to be positioned correctly on those exact dates,
    the annual Sharpe looks great — but it may be mostly luck.

The Monte Carlo stress test randomly shifts each event date by ±N calendar days
(uniformly sampled), rebuilds the `is_event_window` mask, and re-runs the
backtest.  The resulting distribution of Sharpe ratios tells us:

  - Are we capturing a structural pattern, or was timing critical?
  - What is the realistic Sharpe under uncertainty about event timing?
  - What fraction of simulations produce a negative Sharpe? (fragility score)

A "robust" strategy: P&L distribution is tight and centred near the base case.
A "fragile" strategy: most of the P&L comes from one or two event windows;
                      shifting them 2 days wipes out the annual return.

Reference
---------
Harvey, Liu, Zhu (2016) — "… and the Cross-Section of Expected Returns":
formal framework for adjusting Sharpe for the look-ahead embedded in event
timing (extended here to Monte Carlo robustness).
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field

import polars as pl

from ..data.events import BinaryEvent, build_event_mask
from .vectorized import BacktestResult, VectorizedEngine


# --------------------------------------------------------------------------- #
# Result container                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class StressResult:
    """
    Distribution of backtest metrics under random event-date perturbations.

    Attributes
    ----------
    base_sharpe, base_return, base_mdd
        Metrics from the un-perturbed backtest (your baseline).
    sharpe_distribution, return_distribution
        One entry per simulation.  Length == n_simulations.
    p5_sharpe, p95_sharpe
        5th / 95th percentile of the Sharpe distribution.
        Wide gap → fragile (results are sensitive to exact event timing).
    fragility_score
        Fraction of simulations that produced Sharpe < 0.
        0.0 = all sims profitable;  1.0 = all sims lost money.
        A base Sharpe of 1.5 with fragility 0.40 means 40% of the time
        the strategy would have lost money if events had shifted slightly.
    worst_sharpe, best_sharpe
        Min and max across all simulations.
    """
    base_sharpe:          float
    base_return:          float
    base_mdd:             float
    sharpe_distribution:  list[float]
    return_distribution:  list[float]
    p5_sharpe:            float
    p95_sharpe:           float
    fragility_score:      float
    worst_sharpe:         float
    best_sharpe:          float
    n_simulations:        int
    max_shift_days:       int

    def summary(self) -> str:
        w = 60
        bar = "─" * w
        return "\n".join([
            bar,
            f"  Monte Carlo Event Shuffling  ({self.n_simulations:,} simulations)",
            f"  Max date shift: ±{self.max_shift_days} calendar days",
            bar,
            f"  Base Sharpe               {self.base_sharpe:>+8.3f}",
            f"  Base Total Return         {self.base_return:>+8.2%}",
            f"  Base Max Drawdown         {self.base_mdd:>+8.2%}",
            bar,
            f"  Shuffled Sharpe  P5       {self.p5_sharpe:>+8.3f}",
            f"  Shuffled Sharpe  P95      {self.p95_sharpe:>+8.3f}",
            f"  Shuffled Sharpe  Worst    {self.worst_sharpe:>+8.3f}",
            f"  Shuffled Sharpe  Best     {self.best_sharpe:>+8.3f}",
            bar,
            f"  Fragility Score           {self.fragility_score:>8.1%}",
            f"  (fraction of sims with Sharpe < 0)",
            bar,
        ])


# --------------------------------------------------------------------------- #
# EventShuffler                                                                  #
# --------------------------------------------------------------------------- #

class EventShuffler:
    """
    Monte Carlo stress tester: randomly shift event dates and re-run the backtest.

    Parameters
    ----------
    engine : VectorizedEngine
        Pre-configured engine (capital, shares_per_unit, ann_factor, etc.)
    n_simulations : int
        Number of random date-shift draws.  Default 500.
    seed : int
        RNG seed for reproducibility.  Default 42.
    """

    def __init__(
        self,
        engine:       VectorizedEngine,
        n_simulations: int = 500,
        seed:          int = 42,
    ) -> None:
        self.engine        = engine
        self.n_simulations = n_simulations
        self._rng          = random.Random(seed)

    def run(
        self,
        ticker:         str,
        aligned_df:     pl.DataFrame,
        signal_series:  pl.Series,
        events:         list[BinaryEvent],
        max_shift_days: int = 5,
        strategy_name:  str = "unnamed",
    ) -> StressResult:
        """
        Run Monte Carlo simulation over random event-date shifts.

        Parameters
        ----------
        ticker : str
            Ticker symbol (e.g. "NVO").
        aligned_df : pl.DataFrame
            Full aligned DataFrame from DataHandler.align_signals(), BUT
            **without** the 'is_event_window' column (it will be rebuilt each sim).
            Must contain: timestamp, close, log_return, atr_14, signal.
        signal_series : pl.Series
            Pre-computed signal (already shifted 1 bar for PiT).
            This is held constant across simulations — only the event mask changes.
        events : list[BinaryEvent]
            The original event list for this ticker.  Dates will be perturbed.
        max_shift_days : int
            Maximum shift in calendar days (uniform draw from [−max, +max]).
        strategy_name : str
            Label for BacktestResult output.

        Returns
        -------
        StressResult with the full distribution of metrics.
        """
        # ── Baseline (unshifted events) ─────────────────────────────── #
        base_df     = self._rebuild_event_mask(aligned_df, signal_series, events, shift=0)
        base_result = self.engine.run(ticker, base_df, strategy_name)

        # ── Monte Carlo simulations ──────────────────────────────────── #
        sharpe_dist: list[float] = []
        return_dist: list[float] = []

        for _ in range(self.n_simulations):
            shift = self._rng.randint(-max_shift_days, max_shift_days)
            sim_df = self._rebuild_event_mask(aligned_df, signal_series, events, shift)
            try:
                result = self.engine.run(ticker, sim_df, strategy_name)
                sharpe_dist.append(result.net_sharpe)
                return_dist.append(result.total_return)
            except Exception:
                # Degenerate simulation (e.g. all-event window) — skip
                sharpe_dist.append(float("nan"))
                return_dist.append(float("nan"))

        # Remove NaN
        valid_sharpes = [s for s in sharpe_dist if s == s]  # NaN != NaN
        valid_returns = [r for r in return_dist if r == r]

        if not valid_sharpes:
            valid_sharpes = [0.0]

        sorted_s = sorted(valid_sharpes)
        n = len(sorted_s)

        def _pct(data: list[float], p: float) -> float:
            idx = max(0, min(int(p / 100 * (len(data) - 1)), len(data) - 1))
            return data[idx]

        return StressResult(
            base_sharpe         = base_result.net_sharpe,
            base_return         = base_result.total_return,
            base_mdd            = base_result.max_drawdown,
            sharpe_distribution = sharpe_dist,
            return_distribution = return_dist,
            p5_sharpe           = _pct(sorted_s, 5),
            p95_sharpe          = _pct(sorted_s, 95),
            fragility_score     = sum(1 for s in valid_sharpes if s < 0) / max(n, 1),
            worst_sharpe        = sorted_s[0],
            best_sharpe         = sorted_s[-1],
            n_simulations       = self.n_simulations,
            max_shift_days      = max_shift_days,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _rebuild_event_mask(
        self,
        df:            pl.DataFrame,
        signal_series: pl.Series,
        events:        list[BinaryEvent],
        shift:         int,
    ) -> pl.DataFrame:
        """
        Shift all event dates by `shift` calendar days, rebuild the boolean
        is_event_window mask, add it to df alongside the signal series.
        """
        shifted_events = [
            BinaryEvent(
                ticker      = ev.ticker,
                date        = ev.date + dt.timedelta(days=shift),
                event_type  = ev.event_type,
                description = ev.description,
                pre_window  = ev.pre_window,
                post_window = ev.post_window,
            )
            for ev in events
        ]

        # Extract date series from timestamps
        timestamps = df["timestamp"]
        if timestamps.dtype == pl.Datetime:
            date_series = timestamps.cast(pl.Date)
        else:
            date_series = timestamps

        # Build event mask using the shifted dates
        flagged: set[dt.date] = set()
        for ev in shifted_events:
            for delta in range(-ev.pre_window, ev.post_window + 1):
                flagged.add(ev.date + dt.timedelta(days=delta))

        # Pass as Python list to avoid Polars same-dtype is_in deprecation
        event_mask = date_series.is_in(sorted(flagged))

        return df.with_columns([
            event_mask.alias("is_event_window"),
            signal_series.alias("signal"),
        ])
