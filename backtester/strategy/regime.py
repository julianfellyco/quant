"""backtester/strategy/regime.py — Market regime detection.

Regimes classify the current market environment to filter strategy signals.
A momentum strategy should only be active in trending regimes; mean-reversion
strategies work best in ranging regimes.

Detectors add a 'regime' string column to the price DataFrame.
"""
from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

import polars as pl


class Regime(str, Enum):
    """Market regime labels."""

    BULL_TRENDING  = "bull_trending"
    BEAR_TRENDING  = "bear_trending"
    RANGING        = "ranging"
    HIGH_VOL       = "high_vol"


# Strategies that perform well in each regime
STRATEGY_REGIME_MAP: dict[str, set[Regime]] = {
    "momentum":            {Regime.BULL_TRENDING, Regime.BEAR_TRENDING},
    "mean_reversion":      {Regime.RANGING},
    "pairs":               {Regime.RANGING, Regime.BULL_TRENDING},
    "momentum_rank":       {Regime.BULL_TRENDING, Regime.BEAR_TRENDING},
    "mean_reversion_rank": {Regime.RANGING},
}


@runtime_checkable
class RegimeDetector(Protocol):
    """Protocol for regime detectors."""

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add 'regime' column (string) to DataFrame."""
        ...


class SMARegime:
    """Regime detection via SMA crossover + volatility filter.

    Logic:
    1. If realised vol > vol_threshold  → HIGH_VOL (overrides SMA)
    2. If fast SMA > slow SMA           → BULL_TRENDING
    3. If fast SMA < slow SMA           → BEAR_TRENDING
    4. Otherwise (close to crossing)    → RANGING

    Args:
        fast_period:    Short SMA window (default 50 bars).
        slow_period:    Long SMA window (default 200 bars).
        vol_lookback:   Rolling window for realised vol (default 21 bars).
        vol_threshold:  Annualised vol threshold for HIGH_VOL (default 0.30).
    """

    def __init__(
        self,
        fast_period: int = 50,
        slow_period: int = 200,
        vol_lookback: int = 21,
        vol_threshold: float = 0.30,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.vol_lookback = vol_lookback
        self.vol_threshold = vol_threshold

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        """Classify each bar into a market regime and add a 'regime' column.

        Computes a fast SMA, slow SMA, and rolling realised volatility, then
        applies the following priority rules:

        1. Realised vol > vol_threshold  → ``HIGH_VOL`` (overrides SMA signal)
        2. Fast SMA > slow SMA           → ``BULL_TRENDING``
        3. Fast SMA < slow SMA           → ``BEAR_TRENDING``
        4. Otherwise                     → ``RANGING``

        Intermediate columns (_sma_fast, _sma_slow, _realized_vol) are dropped
        before returning.

        Args:
            df: Polars DataFrame containing at minimum a ``close`` column.

        Returns:
            Input DataFrame with an additional ``regime`` string column whose
            values are drawn from the ``Regime`` enum (e.g. ``"bull_trending"``).
        """
        df = df.with_columns([
            pl.col("close").rolling_mean(self.fast_period, min_samples=1).alias("_sma_fast"),
            pl.col("close").rolling_mean(self.slow_period, min_samples=1).alias("_sma_slow"),
            (
                pl.col("close").pct_change()
                .rolling_std(self.vol_lookback, min_samples=2)
                * (252 ** 0.5)
            ).alias("_realized_vol"),
        ])

        df = df.with_columns(
            pl.when(pl.col("_realized_vol") > self.vol_threshold)
            .then(pl.lit(Regime.HIGH_VOL.value))
            .when(pl.col("_sma_fast") > pl.col("_sma_slow"))
            .then(pl.lit(Regime.BULL_TRENDING.value))
            .when(pl.col("_sma_fast") < pl.col("_sma_slow"))
            .then(pl.lit(Regime.BEAR_TRENDING.value))
            .otherwise(pl.lit(Regime.RANGING.value))
            .alias("regime")
        )

        return df.drop(["_sma_fast", "_sma_slow", "_realized_vol"])


def apply_regime_filter(
    df: pl.DataFrame,
    strategy_name: str,
    regime_detector: RegimeDetector,
) -> pl.DataFrame:
    """Zero out signals in unfavourable regimes.

    Applies the regime detector, then sets signal=0 on bars where the
    detected regime is not in the strategy's allowed set.

    Args:
        df:               DataFrame with 'signal' column.
        strategy_name:    Name of strategy (key in STRATEGY_REGIME_MAP).
        regime_detector:  Any object implementing RegimeDetector protocol.

    Returns:
        DataFrame with 'regime' column added and signal filtered.
    """
    if "signal" not in df.columns:
        return df

    df = regime_detector.detect(df)
    allowed = STRATEGY_REGIME_MAP.get(strategy_name, set(Regime))
    allowed_values = [r.value for r in allowed]

    df = df.with_columns(
        pl.when(pl.col("regime").is_in(allowed_values))
        .then(pl.col("signal"))
        .otherwise(pl.lit(0.0))
        .alias("signal")
    )
    return df
