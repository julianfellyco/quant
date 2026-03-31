"""
PaperRunner — generates signals from the backtester engine and submits
confirmed orders to an Alpaca paper trading account.

Safety guarantees (all enforced in _safety_check):
- Paper mode ONLY (LiveConfig.paper_mode is always True)
- Max 100 shares or $10,000 notional per order
- Max 20 orders per calendar day
- Orders only submitted during market hours (9:30–16:00 ET)
- Every order requires explicit caller confirmation via `confirmed=True`
"""
from __future__ import annotations
import datetime as dt
import logging
from dataclasses import dataclass, field

import polars as pl

from backtester.live.broker import Broker, OrderResult
from backtester.live.config import LiveConfig

log = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    ticker: str
    side: str           # "buy" | "sell"
    quantity: int
    strategy: str
    signal_strength: float
    estimated_price: float
    estimated_notional: float
    reasoning: str


@dataclass
class PaperRunnerState:
    orders_today: int = 0
    last_reset_date: dt.date = field(default_factory=dt.date.today)
    order_history: list[OrderResult] = field(default_factory=list)


class PaperRunner:
    """
    Thin bridge between backtester signal generation and Alpaca paper execution.

    Usage:
        runner = PaperRunner(broker, config)
        signal = runner.generate_signal("PFE", "momentum", price_df)
        if signal:
            result = await runner.submit(signal, confirmed=True)
    """

    def __init__(self, broker: Broker, config: LiveConfig | None = None):
        self.broker = broker
        self.config = config or LiveConfig()
        self._state = PaperRunnerState()

    def _reset_daily_counter(self) -> None:
        today = dt.date.today()
        if self._state.last_reset_date != today:
            self._state.orders_today = 0
            self._state.last_reset_date = today

    def _safety_check(self, signal: TradeSignal) -> tuple[bool, str]:
        """Return (allowed, reason). All safety rails enforced here."""
        self._reset_daily_counter()

        if not self.config.paper_mode:
            return False, "Live trading is not supported."

        if not self.config.is_market_hours():
            return False, "Outside market hours (9:30–16:00 ET)."

        if self._state.orders_today >= self.config.max_daily_orders:
            return False, f"Daily order limit ({self.config.max_daily_orders}) reached."

        if signal.quantity > self.config.max_order_shares:
            return False, f"Quantity {signal.quantity} exceeds max {self.config.max_order_shares} shares."

        if signal.estimated_notional > self.config.max_order_notional:
            return False, (
                f"Notional ${signal.estimated_notional:,.0f} exceeds "
                f"max ${self.config.max_order_notional:,.0f}."
            )

        return True, "ok"

    def generate_signal(
        self,
        ticker: str,
        strategy: str,
        price_df: pl.DataFrame,
        capital: float = 100_000.0,
    ) -> TradeSignal | None:
        """
        Generate a trade signal using the backtester's signal functions.
        Returns None if no actionable signal.
        """
        try:
            if strategy == "momentum":
                from backtester.strategy.signals import momentum_signal
                sig_series = momentum_signal(price_df)
            elif strategy == "mean_reversion":
                from backtester.strategy.signals import mean_reversion_signal
                sig_series = mean_reversion_signal(price_df)
            else:
                log.warning("Unknown strategy: %s", strategy)
                return None

            last_signal = float(sig_series[-1])
            if abs(last_signal) < 0.01:
                return None

            last_price = float(price_df["close"][-1])
            side = "buy" if last_signal > 0 else "sell"

            # Conservative sizing: min(100 shares, $10k notional)
            shares = min(
                self.config.max_order_shares,
                int(self.config.max_order_notional / last_price),
            )
            notional = shares * last_price

            return TradeSignal(
                ticker=ticker,
                side=side,
                quantity=shares,
                strategy=strategy,
                signal_strength=last_signal,
                estimated_price=last_price,
                estimated_notional=notional,
                reasoning=f"{strategy} signal={last_signal:.3f} on {ticker} @ ${last_price:.2f}",
            )
        except Exception as exc:
            log.error("Signal generation failed for %s/%s: %s", ticker, strategy, exc)
            return None

    async def submit(self, signal: TradeSignal, confirmed: bool = False) -> OrderResult:
        """
        Submit a paper trade order.

        Args:
            signal: TradeSignal from generate_signal()
            confirmed: Must be explicitly True. Caller must confirm before execution.
        """
        if not confirmed:
            return OrderResult(
                order_id="",
                ticker=signal.ticker,
                side=signal.side,
                quantity=signal.quantity,
                fill_price=None,
                status="rejected",
                error="Order not confirmed. Pass confirmed=True to execute.",
            )

        allowed, reason = self._safety_check(signal)
        if not allowed:
            log.warning("Safety check blocked order for %s: %s", signal.ticker, reason)
            return OrderResult(
                order_id="",
                ticker=signal.ticker,
                side=signal.side,
                quantity=signal.quantity,
                fill_price=None,
                status="rejected",
                error=reason,
            )

        result = await self.broker.submit_order(signal.ticker, signal.quantity, signal.side)
        self._state.orders_today += 1
        self._state.order_history.append(result)
        log.info(
            "Paper order submitted: %s %d %s → status=%s id=%s",
            signal.side, signal.quantity, signal.ticker, result.status, result.order_id,
        )
        return result

    def get_status(self) -> dict:
        self._reset_daily_counter()
        return {
            "orders_today": self._state.orders_today,
            "daily_limit": self.config.max_daily_orders,
            "market_hours": self.config.is_market_hours(),
            "paper_mode": self.config.paper_mode,
            "order_history_count": len(self._state.order_history),
        }
