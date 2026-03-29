"""tests/test_regime.py — Regime detection unit tests."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from backtester.strategy.regime import SMARegime, Regime, apply_regime_filter, STRATEGY_REGIME_MAP


def make_trending_df(n: int = 250, trend: float = 0.5) -> pl.DataFrame:
    """Create uptrending price series."""
    closes = [100.0 + i * trend for i in range(n)]
    return pl.DataFrame({"close": closes})


def make_ranging_df(n: int = 250, amplitude: float = 5.0) -> pl.DataFrame:
    """Create sideways/oscillating price series."""
    t = np.linspace(0, 8 * np.pi, n)
    closes = [100.0 + amplitude * np.sin(x) for x in t]
    return pl.DataFrame({"close": closes})


def make_high_vol_df(n: int = 250) -> pl.DataFrame:
    """Create high-volatility series (large daily moves)."""
    rng = np.random.default_rng(123)
    rets = rng.normal(0, 0.05, n)  # 5% daily vol → 80% annualised
    closes = [100.0]
    for r in rets:
        closes.append(closes[-1] * (1 + r))
    return pl.DataFrame({"close": closes[:n]})


class TestSMARegime:
    def test_produces_regime_column(self):
        df = make_trending_df()
        detector = SMARegime(fast_period=10, slow_period=30, vol_lookback=5)
        result = detector.detect(df)
        assert "regime" in result.columns

    def test_bull_trending_labels(self):
        df = make_trending_df(trend=1.0, n=250)
        detector = SMARegime(fast_period=10, slow_period=50, vol_lookback=5, vol_threshold=0.99)
        result = detector.detect(df)
        # After enough bars, fast SMA > slow SMA in uptrend
        late_regimes = result["regime"].tail(50).to_list()
        assert Regime.BULL_TRENDING.value in late_regimes

    def test_high_vol_overrides_sma(self):
        df = make_high_vol_df(250)
        detector = SMARegime(vol_threshold=0.30, vol_lookback=10)
        result = detector.detect(df)
        regimes = result["regime"].to_list()
        assert Regime.HIGH_VOL.value in regimes

    def test_no_temp_columns_leaked(self):
        df = make_trending_df()
        detector = SMARegime(fast_period=10, slow_period=30)
        result = detector.detect(df)
        assert "_sma_fast" not in result.columns
        assert "_sma_slow" not in result.columns
        assert "_realized_vol" not in result.columns

    def test_all_values_are_valid_regime(self):
        df = make_ranging_df()
        detector = SMARegime()
        result = detector.detect(df)
        valid = {r.value for r in Regime}
        for val in result["regime"].drop_nulls().to_list():
            assert val in valid


class TestApplyRegimeFilter:
    def _df_with_signal(self, n: int = 100, trend: float = 0.5) -> pl.DataFrame:
        base = make_trending_df(n, trend)
        return base.with_columns(pl.lit(1.0).alias("signal"))

    def test_momentum_signals_zeroed_in_ranging(self):
        df = make_ranging_df(250)
        df = df.with_columns(pl.lit(1.0).alias("signal"))
        detector = SMARegime(fast_period=5, slow_period=20, vol_threshold=0.99)
        result = apply_regime_filter(df, "momentum", detector)
        # In ranging regime, momentum signals should be zeroed
        ranging_rows = result.filter(pl.col("regime") == Regime.RANGING.value)
        if not ranging_rows.is_empty():
            assert (ranging_rows["signal"] == 0.0).all()

    def test_mean_reversion_signals_zeroed_in_trending(self):
        df = make_trending_df(250, trend=2.0)
        df = df.with_columns(pl.lit(1.0).alias("signal"))
        detector = SMARegime(fast_period=5, slow_period=20, vol_threshold=0.99)
        result = apply_regime_filter(df, "mean_reversion", detector)
        bull_rows = result.filter(pl.col("regime") == Regime.BULL_TRENDING.value)
        if not bull_rows.is_empty():
            assert (bull_rows["signal"] == 0.0).all()

    def test_unknown_strategy_passes_all_signals(self):
        df = self._df_with_signal(100)
        detector = SMARegime(fast_period=5, slow_period=20)
        result = apply_regime_filter(df, "unknown_strategy_xyz", detector)
        # Unknown strategy: all regimes allowed → no signals zeroed
        assert result["signal"].sum() > 0


class TestStrategyRegimeMap:
    def test_momentum_allows_trending(self):
        assert Regime.BULL_TRENDING in STRATEGY_REGIME_MAP["momentum"]
        assert Regime.BEAR_TRENDING in STRATEGY_REGIME_MAP["momentum"]

    def test_mean_reversion_allows_ranging(self):
        assert Regime.RANGING in STRATEGY_REGIME_MAP["mean_reversion"]

    def test_mean_reversion_blocks_trending(self):
        assert Regime.BULL_TRENDING not in STRATEGY_REGIME_MAP["mean_reversion"]
