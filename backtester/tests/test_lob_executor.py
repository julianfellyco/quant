"""tests/test_lob_executor.py — LOBExecutor unit tests."""
from __future__ import annotations

import pytest
from backtester.engine.lob_executor import LOBExecutor, FillResult


class TestLOBExecutor:
    def setup_method(self):
        self.executor = LOBExecutor(levels=10, base_depth=1_000, tick_size=0.01, depth_decay=0.8)

    def test_zero_quantity_returns_zero_fill_rate(self):
        result = self.executor.simulate_fill(mid_price=100.0, quantity=0)
        assert result.fill_rate == 0.0
        assert result.slippage_bps == 0.0

    def test_buy_fills_above_mid(self):
        result = self.executor.simulate_fill(mid_price=100.0, quantity=100, spread_bps=4.0)
        assert result.fill_price > 100.0

    def test_sell_fills_below_mid(self):
        result = self.executor.simulate_fill(mid_price=100.0, quantity=-100, spread_bps=4.0)
        assert result.fill_price < 100.0

    def test_large_order_higher_slippage(self):
        small = self.executor.simulate_fill(100.0,  100, spread_bps=2.0)
        large = self.executor.simulate_fill(100.0, 5000, spread_bps=2.0)
        assert large.slippage_bps >= small.slippage_bps

    def test_full_fill_small_order(self):
        result = self.executor.simulate_fill(100.0, 500, spread_bps=2.0)
        assert result.fill_rate == 1.0

    def test_partial_fill_huge_order(self):
        executor = LOBExecutor(levels=3, base_depth=100, depth_decay=0.5)
        result = executor.simulate_fill(100.0, 10_000, spread_bps=2.0)
        assert result.fill_rate < 1.0

    def test_cost_usd_nonnegative(self):
        for qty in [-500, -1, 0, 1, 500]:
            cost = self.executor.cost_usd(100.0, qty, spread_bps=3.0)
            assert cost >= 0.0

    def test_cost_usd_zero_for_zero_quantity(self):
        assert self.executor.cost_usd(100.0, 0) == 0.0

    def test_fill_result_is_dataclass(self):
        result = self.executor.simulate_fill(50.0, 200, spread_bps=5.0)
        assert isinstance(result, FillResult)
        assert hasattr(result, "fill_price")
        assert hasattr(result, "slippage_bps")
        assert hasattr(result, "market_impact_bps")
        assert hasattr(result, "fill_rate")
