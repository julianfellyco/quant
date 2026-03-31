"""backtester/tests/test_paper_trading.py — Paper trading unit tests."""
from __future__ import annotations
import asyncio
import datetime as dt
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backtester.live.config import LiveConfig
from backtester.live.broker import OrderResult
from backtester.live.paper_runner import PaperRunner, TradeSignal


def make_signal(qty: int = 50, notional: float = 5000.0) -> TradeSignal:
    return TradeSignal(
        ticker="PFE",
        side="buy",
        quantity=qty,
        strategy="momentum",
        signal_strength=0.8,
        estimated_price=notional / qty,
        estimated_notional=notional,
        reasoning="test signal",
    )


def make_runner(market_hours: bool = True) -> tuple[PaperRunner, MagicMock]:
    broker = MagicMock()
    broker.submit_order = AsyncMock(return_value=OrderResult(
        order_id="abc123", ticker="PFE", side="buy", quantity=50,
        fill_price=35.0, status="filled",
    ))
    config = LiveConfig()
    runner = PaperRunner(broker=broker, config=config)
    runner.config.is_market_hours = lambda: market_hours
    return runner, broker


class TestLiveConfig:
    def test_paper_mode_always_true(self):
        config = LiveConfig()
        assert config.paper_mode is True

    def test_max_order_limits(self):
        config = LiveConfig()
        assert config.max_order_shares == 100
        assert config.max_order_notional == 10_000.0
        assert config.max_daily_orders == 20

    def test_live_url_rejected(self):
        from backtester.live.broker import AlpacaPaperBroker
        with pytest.raises(ValueError, match="Live trading is not supported"):
            AlpacaPaperBroker("key", "secret", base_url="https://api.alpaca.markets")


class TestPaperRunner:
    def test_unconfirmed_order_rejected(self):
        runner, broker = make_runner()
        signal = make_signal()
        result = asyncio.run(runner.submit(signal, confirmed=False))
        assert result.status == "rejected"
        assert "confirmed" in (result.error or "").lower()
        broker.submit_order.assert_not_called()

    def test_outside_market_hours_blocked(self):
        runner, broker = make_runner(market_hours=False)
        signal = make_signal()
        result = asyncio.run(runner.submit(signal, confirmed=True))
        assert result.status == "rejected"
        assert "market hours" in (result.error or "").lower()
        broker.submit_order.assert_not_called()

    def test_oversized_shares_blocked(self):
        runner, broker = make_runner()
        signal = make_signal(qty=150, notional=5250.0)
        result = asyncio.run(runner.submit(signal, confirmed=True))
        assert result.status == "rejected"
        assert "shares" in (result.error or "").lower()

    def test_oversized_notional_blocked(self):
        runner, broker = make_runner()
        signal = make_signal(qty=50, notional=15_000.0)
        result = asyncio.run(runner.submit(signal, confirmed=True))
        assert result.status == "rejected"
        assert "notional" in (result.error or "").lower() or "10,000" in (result.error or "")

    def test_daily_order_limit(self):
        runner, broker = make_runner()
        runner._state.orders_today = 20
        signal = make_signal()
        result = asyncio.run(runner.submit(signal, confirmed=True))
        assert result.status == "rejected"
        assert "limit" in (result.error or "").lower()

    def test_valid_order_passes_through(self):
        runner, broker = make_runner(market_hours=True)
        signal = make_signal(qty=50, notional=1750.0)
        result = asyncio.run(runner.submit(signal, confirmed=True))
        assert result.status == "filled"
        assert runner._state.orders_today == 1

    def test_daily_counter_resets(self):
        runner, _ = make_runner()
        runner._state.orders_today = 5
        runner._state.last_reset_date = dt.date(2000, 1, 1)  # old date
        runner._reset_daily_counter()
        assert runner._state.orders_today == 0

    def test_get_status_returns_dict(self):
        runner, _ = make_runner()
        status = runner.get_status()
        assert "orders_today" in status
        assert "paper_mode" in status
        assert status["paper_mode"] is True
