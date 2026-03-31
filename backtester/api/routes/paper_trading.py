"""backtester/api/routes/paper_trading.py — Paper trading API endpoints."""
from __future__ import annotations
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backtester.live.config import LiveConfig
from backtester.live.paper_runner import PaperRunner, TradeSignal
from backtester.live.broker import AlpacaPaperBroker, OrderResult

log = logging.getLogger(__name__)
router = APIRouter()

# Module-level runner (initialized lazily)
_runner: PaperRunner | None = None


def _get_runner() -> PaperRunner:
    global _runner
    if _runner is None:
        config = LiveConfig()
        broker = AlpacaPaperBroker(
            api_key=config.api_key,
            secret_key=config.secret_key,
        )
        _runner = PaperRunner(broker=broker, config=config)
    return _runner


class PaperTradeRequest(BaseModel):
    ticker: str
    strategy: Literal["momentum", "mean_reversion"]
    action: Literal["signal", "submit", "status"]
    confirmed: bool = False          # must be True to actually submit
    lookback_days: int = 90


class PaperTradeResponse(BaseModel):
    action: str
    ticker: str | None = None
    status: str | None = None
    signal: dict | None = None
    order: dict | None = None
    runner_status: dict | None = None
    error: str | None = None


@router.post("/paper-trade", response_model=PaperTradeResponse)
async def paper_trade(req: PaperTradeRequest) -> PaperTradeResponse:
    """
    Paper trading endpoint.

    Actions:
    - "signal": generate a signal without submitting
    - "submit": generate and submit (requires confirmed=True)
    - "status": return runner state
    """
    runner = _get_runner()

    if req.action == "status":
        return PaperTradeResponse(action="status", runner_status=runner.get_status())

    # Fetch price data for signal generation
    import datetime as dt
    from backtester.data.handler import DataHandler, Granularity
    try:
        end = dt.date.today()
        start = end - dt.timedelta(days=int(req.lookback_days * 1.5))
        handler = DataHandler([req.ticker], granularity=Granularity.DAILY)
        handler.load(start=start, end=end)
        price_df = handler[req.ticker]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Data fetch failed: {exc}")

    signal = runner.generate_signal(req.ticker, req.strategy, price_df)
    if signal is None:
        return PaperTradeResponse(
            action=req.action,
            ticker=req.ticker,
            status="no_signal",
            signal=None,
        )

    signal_dict = {
        "ticker": signal.ticker,
        "side": signal.side,
        "quantity": signal.quantity,
        "strategy": signal.strategy,
        "signal_strength": signal.signal_strength,
        "estimated_price": signal.estimated_price,
        "estimated_notional": signal.estimated_notional,
        "reasoning": signal.reasoning,
    }

    if req.action == "signal":
        return PaperTradeResponse(action="signal", ticker=req.ticker, status="ok", signal=signal_dict)

    if req.action == "submit":
        result: OrderResult = await runner.submit(signal, confirmed=req.confirmed)
        return PaperTradeResponse(
            action="submit",
            ticker=req.ticker,
            status=result.status,
            signal=signal_dict,
            order={
                "order_id": result.order_id,
                "side": result.side,
                "quantity": result.quantity,
                "fill_price": result.fill_price,
                "status": result.status,
                "error": result.error,
            },
        )

    raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")
