"""tests/test_risk.py — Unit tests for the risk module."""
from __future__ import annotations

import pytest
import polars as pl

from backtester.risk.position_sizer import FixedFractional, VolatilityTarget, KellyCriterion
from backtester.risk.stop_loss import ATRStop, FixedPercentStop
from backtester.risk.portfolio_risk import RiskLimits
from backtester.risk.kelly import kelly_fraction, fractional_kelly, kelly_from_sharpe


# ── FixedFractional ────────────────────────────────────────────────────────── #

class TestFixedFractional:
    def test_basic_size(self):
        sizer = FixedFractional(risk_pct=0.02, max_position_pct=0.20)
        size = sizer.compute_size(
            capital=100_000, price=50.0, signal_strength=1.0,
            volatility=0.20, adv=1_000_000,
        )
        # With 2% risk, 2× daily vol stop: risk = 2000, stop = 50×(0.2/√252)×2 ≈ 1.26
        # raw_size = 2000 / 1.26 ≈ 1587; capped by adv_limit = 10000
        assert size > 0

    def test_zero_price_returns_zero(self):
        sizer = FixedFractional()
        assert sizer.compute_size(100_000, 0.0, 1.0, 0.20, 1_000_000) == 0

    def test_max_position_cap(self):
        sizer = FixedFractional(risk_pct=0.02, max_position_pct=0.05)
        size = sizer.compute_size(
            capital=100_000, price=10.0, signal_strength=1.0,
            volatility=0.01, adv=10_000_000,  # low vol → huge raw size
        )
        # max_position_pct cap: 5000 / 10 = 500 shares
        assert size <= 500

    def test_adv_cap(self):
        sizer = FixedFractional(risk_pct=0.02, max_position_pct=0.50)
        size = sizer.compute_size(
            capital=100_000, price=10.0, signal_strength=1.0,
            volatility=0.01, adv=100,  # tiny ADV → adv_limit = 1 share
        )
        assert size <= 1

    def test_negative_signal(self):
        sizer = FixedFractional()
        size = sizer.compute_size(100_000, 50.0, -1.0, 0.20, 1_000_000)
        assert size <= 0


# ── VolatilityTarget ───────────────────────────────────────────────────────── #

class TestVolatilityTarget:
    def test_scales_with_vol(self):
        sizer = VolatilityTarget(target_vol=0.15, max_leverage=1.0)
        size_low_vol = sizer.compute_size(100_000, 50.0, 1.0, 0.10, 10_000_000)
        size_high_vol = sizer.compute_size(100_000, 50.0, 1.0, 0.30, 10_000_000)
        # Lower vol → larger position (more shares needed to hit target vol)
        assert size_low_vol > size_high_vol

    def test_max_leverage_respected(self):
        sizer = VolatilityTarget(target_vol=0.15, max_leverage=0.5)
        size = sizer.compute_size(100_000, 10.0, 1.0, 0.01, 100_000_000)
        # max notional = 100_000 × 0.5 = 50_000 → max shares = 5000
        assert size <= 5000

    def test_zero_volatility_returns_zero(self):
        sizer = VolatilityTarget()
        assert sizer.compute_size(100_000, 50.0, 1.0, 0.0, 1_000_000) == 0


# ── KellyCriterion ─────────────────────────────────────────────────────────── #

class TestKellyCriterion:
    def test_positive_edge(self):
        sizer = KellyCriterion(fraction=0.25, win_rate=0.55, avg_win_loss_ratio=1.5)
        size = sizer.compute_size(100_000, 50.0, 1.0, 0.20, 10_000_000)
        assert size > 0

    def test_zero_edge_returns_zero(self):
        # win_rate=0.40, R=1.0 → Kelly = 0.4 - 0.6/1.0 = -0.2 → clamped to 0
        sizer = KellyCriterion(fraction=0.25, win_rate=0.40, avg_win_loss_ratio=1.0)
        size = sizer.compute_size(100_000, 50.0, 1.0, 0.20, 10_000_000)
        assert size == 0

    def test_max_position_cap(self):
        sizer = KellyCriterion(fraction=1.0, max_position_pct=0.10, win_rate=0.70, avg_win_loss_ratio=3.0)
        size = sizer.compute_size(100_000, 10.0, 1.0, 0.20, 10_000_000)
        # max notional = 10_000 → max shares = 1000
        assert size <= 1000


# ── ATRStop ────────────────────────────────────────────────────────────────── #

class TestATRStop:
    def _sample_df(self) -> pl.DataFrame:
        n = 30
        closes = [100.0 + i * 0.5 for i in range(n)]
        return pl.DataFrame({
            "close": closes,
            "high":  [c + 1.0 for c in closes],
            "low":   [c - 1.0 for c in closes],
        })

    def test_stop_price_column_exists(self):
        stop = ATRStop(atr_period=5, atr_multiplier=2.0)
        df = stop.apply(self._sample_df())
        assert "stop_price" in df.columns

    def test_stop_price_below_close(self):
        stop = ATRStop(atr_period=5, atr_multiplier=2.0)
        df = stop.apply(self._sample_df())
        # stop_price should be below close (long stop)
        valid = df.filter(pl.col("stop_price").is_not_null())
        assert (valid["stop_price"] < valid["close"]).all()

    def test_no_extra_columns(self):
        stop = ATRStop(atr_period=5, atr_multiplier=2.0)
        df = self._sample_df()
        result = stop.apply(df)
        # _atr_stop temp column must be cleaned up
        assert "_atr_stop" not in result.columns


# ── FixedPercentStop ───────────────────────────────────────────────────────── #

class TestFixedPercentStop:
    def test_stop_is_correct_pct(self):
        stop = FixedPercentStop(stop_pct=0.05)
        df = pl.DataFrame({"close": [100.0, 110.0, 90.0]})
        result = stop.apply(df)
        assert "stop_price" in result.columns
        expected = [95.0, 104.5, 85.5]
        for actual, exp in zip(result["stop_price"].to_list(), expected):
            assert abs(actual - exp) < 0.01


# ── RiskLimits ─────────────────────────────────────────────────────────────── #

class TestRiskLimits:
    def test_allows_normal_trade(self):
        limits = RiskLimits(max_single_position=0.20)
        allowed, msg = limits.check_new_trade({}, "AAPL", 10_000, 100_000)
        assert allowed
        assert msg == "ok"

    def test_blocks_oversized_position(self):
        limits = RiskLimits(max_single_position=0.10)
        allowed, msg = limits.check_new_trade({}, "AAPL", 20_000, 100_000)
        assert not allowed
        assert "20.0%" in msg or "max" in msg.lower()

    def test_blocks_over_100pct_exposure(self):
        limits = RiskLimits()
        existing = {"SPY": 95_000.0}
        allowed, msg = limits.check_new_trade(existing, "AAPL", 20_000, 100_000)
        assert not allowed
        assert "100%" in msg or "exposure" in msg.lower()

    def test_drawdown_halt(self):
        limits = RiskLimits(max_drawdown_halt=0.15)
        ok, _ = limits.check_drawdown(-0.10)
        assert ok
        halt, msg = limits.check_drawdown(-0.16)
        assert not halt
        assert "halt" in msg.lower() or "drawdown" in msg.lower()


# ── Kelly utilities ────────────────────────────────────────────────────────── #

class TestKellyUtilities:
    def test_kelly_fraction_positive(self):
        f = kelly_fraction(win_rate=0.55, avg_win=1.5, avg_loss=1.0)
        assert 0 < f < 1

    def test_kelly_fraction_negative_edge(self):
        f = kelly_fraction(win_rate=0.40, avg_win=1.0, avg_loss=1.0)
        assert f == 0.0  # clamped

    def test_fractional_kelly_is_fraction(self):
        full = kelly_fraction(0.55, 1.5, 1.0)
        frac = fractional_kelly(0.55, 1.5, 1.0, fraction=0.25)
        assert abs(frac - full * 0.25) < 1e-9

    def test_kelly_from_sharpe(self):
        f = kelly_from_sharpe(sharpe=1.0, ann_factor=252)
        assert f > 0
        assert f < 0.1  # per-bar Kelly for Sharpe=1 daily is tiny
