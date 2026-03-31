"""Broker protocol and Alpaca paper trading implementation."""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger(__name__)


@dataclass
class OrderResult:
    order_id: str
    ticker: str
    side: str           # "buy" | "sell"
    quantity: int
    fill_price: float | None
    status: str         # "filled" | "pending" | "rejected"
    error: str | None = None


class Broker(Protocol):
    async def submit_order(self, ticker: str, quantity: int, side: str) -> OrderResult: ...
    async def get_positions(self) -> dict[str, float]: ...
    async def get_account(self) -> dict: ...


class AlpacaPaperBroker:
    """
    Alpaca paper trading broker.

    Only paper-api.alpaca.markets is ever used.
    Requires alpaca-py: pip install alpaca-py
    """

    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets"):
        if "live" in base_url or "api.alpaca.markets" == base_url.rstrip("/").split("//")[-1]:
            raise ValueError("Live trading is not supported. Use paper-api.alpaca.markets only.")
        self._api_key = api_key
        self._secret_key = secret_key
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from alpaca.trading.client import TradingClient
                self._client = TradingClient(
                    api_key=self._api_key,
                    secret_key=self._secret_key,
                    paper=True,
                )
            except ImportError as e:
                raise ImportError("alpaca-py not installed. Run: pip install alpaca-py") from e
        return self._client

    async def submit_order(self, ticker: str, quantity: int, side: str) -> OrderResult:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        try:
            client = self._get_client()
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            req = MarketOrderRequest(
                symbol=ticker,
                qty=quantity,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            order = client.submit_order(req)
            return OrderResult(
                order_id=str(order.id),
                ticker=ticker,
                side=side,
                quantity=quantity,
                fill_price=float(order.filled_avg_price) if order.filled_avg_price else None,
                status=str(order.status),
            )
        except Exception as exc:
            log.error("Order failed for %s: %s", ticker, exc)
            return OrderResult(
                order_id="",
                ticker=ticker,
                side=side,
                quantity=quantity,
                fill_price=None,
                status="rejected",
                error=str(exc),
            )

    async def get_positions(self) -> dict[str, float]:
        try:
            client = self._get_client()
            positions = client.get_all_positions()
            return {p.symbol: float(p.market_value) for p in positions}
        except Exception as exc:
            log.error("get_positions failed: %s", exc)
            return {}

    async def get_account(self) -> dict:
        try:
            client = self._get_client()
            acct = client.get_account()
            return {
                "equity": float(acct.equity),
                "cash": float(acct.cash),
                "buying_power": float(acct.buying_power),
                "portfolio_value": float(acct.portfolio_value),
                "paper_mode": True,
            }
        except Exception as exc:
            log.error("get_account failed: %s", exc)
            return {"error": str(exc), "paper_mode": True}
