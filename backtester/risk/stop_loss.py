"""backtester/risk/stop_loss.py — Stop-loss strategies as Polars transformations."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class StopLoss(Protocol):
    """Protocol for stop-loss strategies."""

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add 'stop_price' column to DataFrame."""
        ...


class ATRStop:
    """ATR-based trailing stop.

    stop_price = close - atr_multiplier × ATR(atr_period)

    For long positions the stop ratchets up with the rolling max of
    the stop price (trailing).  The `apply()` method adds the stop_price
    column; the engine checks whether close < stop_price to trigger exit.
    """

    def __init__(self, atr_period: int = 14, atr_multiplier: float = 2.0) -> None:
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        high = pl.col("high") if "high" in df.columns else pl.col("close")
        low  = pl.col("low")  if "low"  in df.columns else pl.col("close")
        prev_close = pl.col("close").shift(1)

        tr = pl.max_horizontal(
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        )
        df = df.with_columns(
            tr.rolling_mean(self.atr_period, min_samples=1).alias("_atr_stop"),
        )
        df = df.with_columns(
            (pl.col("close") - pl.col("_atr_stop") * self.atr_multiplier).alias("stop_price"),
        )
        return df.drop("_atr_stop")


class FixedPercentStop:
    """Fixed-percentage stop-loss from entry price.

    stop_price = entry_price × (1 − stop_pct)

    Requires an 'entry_price' column (set when position opens).
    Falls back to close × (1 − stop_pct) if entry_price is absent.
    """

    def __init__(self, stop_pct: float = 0.05) -> None:
        self.stop_pct = stop_pct

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        base = pl.col("entry_price") if "entry_price" in df.columns else pl.col("close")
        df = df.with_columns(
            (base * (1.0 - self.stop_pct)).alias("stop_price"),
        )
        return df
