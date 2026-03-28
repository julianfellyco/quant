"""api/routes/tickers.py — GET /api/tickers."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backtester.data.events import EVENTS, EventType
from backtester.engine.costs import COST_PARAMS

router = APIRouter()

_TICKER_META = {
    "NVO": {
        "name":              "Novo Nordisk ADR",
        "exchange":          "NYSE",
        "approx_price_2024": 110.0,
    },
    "PFE": {
        "name":              "Pfizer Inc.",
        "exchange":          "NYSE",
        "approx_price_2024": 27.0,
    },
}


class EventOut(BaseModel):
    date:        str
    event_type:  str
    description: str
    pre_window:  int
    post_window: int


class TickerInfo(BaseModel):
    ticker:            str
    name:              str
    exchange:          str
    avg_daily_volume:  int
    approx_price_2024: float
    base_bps:          float
    events:            list[EventOut]


class TickersResponse(BaseModel):
    tickers: list[TickerInfo]


@router.get("/tickers", response_model=TickersResponse)
def list_tickers() -> TickersResponse:
    tickers = []
    for t, meta in _TICKER_META.items():
        p = COST_PARAMS[t]
        events = [
            EventOut(
                date        = ev.date.isoformat(),
                event_type  = ev.event_type.name,
                description = ev.description,
                pre_window  = ev.pre_window,
                post_window = ev.post_window,
            )
            for ev in EVENTS
            if ev.ticker == t
        ]
        tickers.append(TickerInfo(
            ticker            = t,
            name              = meta["name"],
            exchange          = meta["exchange"],
            avg_daily_volume  = p["base_liquidity"],
            approx_price_2024 = meta["approx_price_2024"],
            base_bps          = p["base_bps"],
            events            = events,
        ))
    return TickersResponse(tickers=tickers)
