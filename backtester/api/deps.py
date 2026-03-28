"""api/deps.py — Shared factories and helpers for route handlers."""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
from typing import Any

from backtester.data.handler import DataHandler, Granularity
from backtester.engine.vectorized import VectorizedEngine, BacktestResult
from backtester.stats.metrics import EventWindowReport, WindowStats


_GRAN = {
    "daily":  Granularity.DAILY,
    "hour":   Granularity.HOUR,
    "minute": Granularity.MINUTE,
}

_ANN = {
    "daily":  252,
    "hour":   1764,
    "minute": 98_280,
}


def granularity_enum(s: str) -> Granularity:
    try:
        return _GRAN[s.lower()]
    except KeyError:
        raise ValueError(f"Unknown granularity '{s}'. Choose from: {list(_GRAN)}")


def make_engine(
    initial_capital: float = 100_000.0,
    shares_per_unit: int   = 1_000,
    risk_free_rate:  float = 0.05,
    granularity:     str   = "daily",
) -> VectorizedEngine:
    return VectorizedEngine(
        initial_capital = initial_capital,
        shares_per_unit = shares_per_unit,
        risk_free_rate  = risk_free_rate,
        ann_factor      = _ANN.get(granularity.lower(), 252),
    )


def ts_to_str(val: Any) -> str:
    """Convert any timestamp variant (datetime, date, str) to ISO string."""
    if isinstance(val, (dt.datetime, dt.date)):
        return val.isoformat()
    return str(val)


def serialize_equity_curve(result: BacktestResult) -> list[dict]:
    """Convert a BacktestResult equity_curve DataFrame to JSON-safe dicts."""
    rows = result.equity_curve.rename({"transaction_cost_usd": "cost_usd"}).to_dicts()
    for row in rows:
        row["timestamp"] = ts_to_str(row["timestamp"])
        # Replace NaN/inf with None so JSON serialization doesn't fail
        for k, v in row.items():
            if isinstance(v, float) and not math.isfinite(v):
                row[k] = None
    return rows


def serialize_window_stats(ws: WindowStats) -> dict:
    d = dataclasses.asdict(ws)
    for k, v in d.items():
        if isinstance(v, float) and not math.isfinite(v):
            d[k] = None
    return d


def clean_float(v: Any) -> Any:
    """Replace non-finite floats with None for JSON safety."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v
