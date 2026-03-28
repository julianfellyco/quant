"""
engine/walkforward.py — Rolling walk-forward parameter optimisation.

Why walk-forward matters
--------------------------
Static parameters (entry_z = 1.5, exit_z = 0.3) are chosen by fitting on
historical data.  If the same data is used to choose AND evaluate parameters,
the in-sample Sharpe will always be unrealistically high — the parameters are
overfit to past noise.

Walk-forward validation addresses this with a rolling train/test protocol:

    Fold 0: optimise on [t0, t0+train) → evaluate on [t0+train, t0+train+test)
    Fold 1: optimise on [t1, t1+train) → evaluate on [t1+train, t1+train+test)
    ...

The out-of-sample (OOS) Sharpes from each fold are the best available
estimate of live performance.

Key metrics
-----------
  aggregate_oos_sharpe   Mean of per-fold OOS Sharpes.
                         This is the honest estimate of expected live Sharpe.
  stability_score        Std dev of per-fold OOS Sharpes.
                         Low → strategy behaviour is consistent across time.
                         High → the edge is regime-dependent (dangerous).
  best_params_per_fold   Which parameters were optimal on each training window.
                         If these vary wildly across folds, the strategy is
                         fitting noise rather than structure.

Usage
-----
>>> from backtester.engine.walkforward import WalkForwardOptimizer
>>> from backtester.engine.vectorized import VectorizedEngine
>>> from backtester.strategy.signals import mean_reversion_signal
>>>
>>> engine = VectorizedEngine(initial_capital=100_000, ann_factor=252)
>>> optimizer = WalkForwardOptimizer(
...     engine=engine,
...     signal_fn=mean_reversion_signal,
...     param_grid={"entry_z": [1.0, 1.5, 2.0], "exit_z": [0.3, 0.5]},
...     train_bars=120,   # ~6 months of daily data
...     test_bars=20,     # ~1 month of daily data
... )
>>> result = optimizer.run("PFE", aligned_df)
>>> print(result.summary())
"""

from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

import polars as pl

from .vectorized import BacktestResult, VectorizedEngine


# --------------------------------------------------------------------------- #
# Result containers                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class FoldResult:
    """Metrics for a single walk-forward fold."""
    fold_index:    int
    train_start:   Any    # timestamp value
    train_end:     Any
    test_start:    Any
    test_end:      Any
    best_params:   dict[str, Any]
    is_sharpe:     float   # in-sample Sharpe (on training window)
    oos_sharpe:    float   # out-of-sample Sharpe (on test window)
    oos_return:    float
    oos_mdd:       float


@dataclass
class WalkForwardResult:
    """
    Aggregate results from all walk-forward folds.

    Attributes
    ----------
    folds : list[FoldResult]
        Per-fold detailed results.
    aggregate_oos_sharpe : float
        Mean of OOS Sharpes — the honest performance estimate.
    is_sharpe_mean : float
        Mean of in-sample Sharpes.  If much larger than OOS, suspect overfitting.
    stability_score : float
        Std dev of OOS Sharpes.  Lower is better (more consistent behaviour).
    sharpe_degradation : float
        is_sharpe_mean − aggregate_oos_sharpe.  Positive = some IS→OOS decay.
    best_params_per_fold : list[dict]
        Optimal parameters per fold (check for stability across time).
    """
    folds:                  list[FoldResult]
    aggregate_oos_sharpe:   float
    is_sharpe_mean:         float
    stability_score:        float
    sharpe_degradation:     float
    best_params_per_fold:   list[dict[str, Any]]
    n_folds:                int

    def summary(self) -> str:
        w = 62
        bar = "─" * w
        lines = [
            bar,
            f"  Walk-Forward Optimisation  ({self.n_folds} folds)",
            bar,
            f"  IS  Sharpe (mean)          {self.is_sharpe_mean:>+8.3f}",
            f"  OOS Sharpe (mean)          {self.aggregate_oos_sharpe:>+8.3f}",
            f"  Sharpe Degradation (IS−OOS){self.sharpe_degradation:>+8.3f}",
            f"  Stability Score (OOS std)  {self.stability_score:>8.3f}",
            bar,
            "  Per-fold OOS Sharpes:",
        ]
        for fold in self.folds:
            lines.append(
                f"    Fold {fold.fold_index:2d}  "
                f"[test {fold.test_start} → {fold.test_end}]  "
                f"OOS Sharpe = {fold.oos_sharpe:+.3f}  "
                f"params = {fold.best_params}"
            )
        lines.append(bar)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# WalkForwardOptimizer                                                           #
# --------------------------------------------------------------------------- #

class WalkForwardOptimizer:
    """
    Rolling walk-forward parameter optimiser.

    Parameters
    ----------
    engine : VectorizedEngine
        Pre-configured backtesting engine.
    signal_fn : Callable
        Function signature: (df: pl.DataFrame, **params) -> pl.Series
        Returns a raw (un-shifted) signal series.
        Compatible with mean_reversion_signal and momentum_signal.
    param_grid : dict[str, list]
        Parameter search space.  All combinations are evaluated on each
        training window.  Example:
            {"entry_z": [1.0, 1.5, 2.0], "exit_z": [0.3, 0.5]}
        Produces 6 candidate parameter sets per fold.
    train_bars : int
        Number of bars in the training window.  Default 120 (≈6 months daily).
    test_bars : int
        Number of bars in the test (OOS) window.  Default 21 (≈1 month daily).
    step_bars : int
        How many bars to advance the window for each fold.  Default = test_bars
        (non-overlapping test periods).
    optimise_on : str
        Metric to maximise on the training window.  "sharpe" (default) or
        "sortino".  Sharpe is faster; Sortino penalises downside more.
    """

    def __init__(
        self,
        engine:       VectorizedEngine,
        signal_fn:    Callable,
        param_grid:   dict[str, list],
        train_bars:   int = 120,
        test_bars:    int = 21,
        step_bars:    int | None = None,
        optimise_on:  str = "sharpe",
    ) -> None:
        self.engine      = engine
        self.signal_fn   = signal_fn
        self.param_grid  = param_grid
        self.train_bars  = train_bars
        self.test_bars   = test_bars
        self.step_bars   = step_bars if step_bars is not None else test_bars
        self.optimise_on = optimise_on

        # Pre-compute all parameter combinations
        keys   = list(param_grid.keys())
        values = list(param_grid.values())
        self._param_combos: list[dict[str, Any]] = [
            dict(zip(keys, combo))
            for combo in itertools.product(*values)
        ]

    # ------------------------------------------------------------------ #
    # Entry point                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        ticker:        str,
        df:            pl.DataFrame,
        strategy_name: str = "unnamed",
    ) -> WalkForwardResult:
        """
        Execute walk-forward optimisation on the full DataFrame.

        Parameters
        ----------
        ticker : str
            Ticker symbol, passed to the engine cost model.
        df : pl.DataFrame
            Full aligned DataFrame (timestamp, close, log_return, atr_14,
            and any optional columns).  Must NOT have a 'signal' column yet.
        strategy_name : str
            Label for result summaries.

        Returns
        -------
        WalkForwardResult with per-fold details and aggregate OOS metrics.
        """
        if "signal" in df.columns:
            df = df.drop("signal")

        folds = self._generate_folds(df)
        if not folds:
            raise ValueError(
                f"DataFrame has {len(df)} bars but needs at least "
                f"{self.train_bars + self.test_bars} for one fold."
            )

        fold_results: list[FoldResult] = []

        for i, (train_df, test_df) in enumerate(folds):
            best_params, is_sharpe = self._optimise_fold(ticker, train_df)

            oos_result = self._evaluate(ticker, test_df, best_params, strategy_name)

            fold_results.append(FoldResult(
                fold_index  = i,
                train_start = train_df["timestamp"][0],
                train_end   = train_df["timestamp"][-1],
                test_start  = test_df["timestamp"][0],
                test_end    = test_df["timestamp"][-1],
                best_params = best_params,
                is_sharpe   = is_sharpe,
                oos_sharpe  = oos_result.net_sharpe,
                oos_return  = oos_result.total_return,
                oos_mdd     = oos_result.max_drawdown,
            ))

        oos_sharpes = [f.oos_sharpe for f in fold_results]
        is_sharpes  = [f.is_sharpe  for f in fold_results]

        agg_oos  = statistics.mean(oos_sharpes)
        agg_is   = statistics.mean(is_sharpes)
        stab     = statistics.stdev(oos_sharpes) if len(oos_sharpes) > 1 else 0.0

        return WalkForwardResult(
            folds                 = fold_results,
            aggregate_oos_sharpe  = agg_oos,
            is_sharpe_mean        = agg_is,
            stability_score       = stab,
            sharpe_degradation    = agg_is - agg_oos,
            best_params_per_fold  = [f.best_params for f in fold_results],
            n_folds               = len(fold_results),
        )

    # ------------------------------------------------------------------ #
    # Fold generation                                                       #
    # ------------------------------------------------------------------ #

    def _generate_folds(
        self, df: pl.DataFrame
    ) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Generate rolling train/test splits.

        Returns a list of (train_df, test_df) pairs.  The test windows are
        non-overlapping (each bar appears in at most one test window).
        """
        n = len(df)
        folds: list[tuple[pl.DataFrame, pl.DataFrame]] = []

        train_end = self.train_bars

        while train_end + self.test_bars <= n:
            test_end   = train_end + self.test_bars
            train_df   = df.slice(train_end - self.train_bars, self.train_bars)
            test_df    = df.slice(train_end, self.test_bars)

            if len(train_df) >= self.train_bars and len(test_df) >= 1:
                folds.append((train_df, test_df))

            train_end += self.step_bars

        return folds

    # ------------------------------------------------------------------ #
    # In-sample optimisation                                               #
    # ------------------------------------------------------------------ #

    def _optimise_fold(
        self,
        ticker:   str,
        train_df: pl.DataFrame,
    ) -> tuple[dict[str, Any], float]:
        """
        Exhaustive grid search on the training window.

        Returns (best_params, best_metric_value).
        """
        best_params: dict[str, Any] = {}
        best_metric: float          = float("-inf")

        for params in self._param_combos:
            try:
                result = self._evaluate(ticker, train_df, params, strategy_name="wf_train")
                metric = result.net_sharpe if self.optimise_on == "sharpe" else result.sortino
                if math.isfinite(metric) and metric > best_metric:
                    best_metric = metric
                    best_params = params
            except Exception:
                continue

        if not best_params:
            # Fallback: use first combo if all failed
            best_params = self._param_combos[0] if self._param_combos else {}

        return best_params, best_metric

    # ------------------------------------------------------------------ #
    # Single evaluation                                                     #
    # ------------------------------------------------------------------ #

    def _evaluate(
        self,
        ticker:        str,
        df:            pl.DataFrame,
        params:        dict[str, Any],
        strategy_name: str,
    ) -> BacktestResult:
        """Generate signal with params, shift by 1 bar, run engine."""
        raw_signal = self.signal_fn(df, **params)

        # PiT enforcement: shift signal by 1 bar before execution
        aligned_df = df.with_columns(
            raw_signal.shift(1).fill_null(0.0).alias("signal")
        )

        return self.engine.run(ticker, aligned_df, strategy_name)
