"""
data/events.py — Binary event calendar for pharma-specific backtesting.

Quant Why: Pharma equities are "jump diffusion" processes, not pure Brownian
motion. Standard volatility models (GARCH, historical vol) systematically
underestimate risk around FDA decisions and clinical readouts because those
events introduce a discrete, bimodal jump:
  - Approval  → +20–40% gap-up
  - Rejection → −30–60% gap-down

Treating these windows the same as normal trading days produces:
  1. Overfit Sharpe ratios (strategy "works" because it happened to be
     correctly positioned on binary-outcome days)
  2. Misleading Max Drawdown figures (a single FDA rejection wipes months
     of P&L but looks like a 1-day event in a standard MDD chart)

This module tags known binary events so the Statistics module can:
  - Exclude them from core Sharpe / Sortino calculation
  - Report a separate "event-window P&L" as a stress test metric
  - Optionally flatten positions N days before known events
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Sequence


class EventType(Enum):
    FDA_DECISION    = auto()   # PDUFA date / approval ruling
    CLINICAL_TRIAL  = auto()   # Phase II / III readout
    EARNINGS        = auto()   # Quarterly earnings release
    CONFERENCE      = auto()   # Major medical conference (ASH, ASCO, ADA)
    MACRO           = auto()   # Non-company event (Fed, CPI) — lower impact


@dataclass(frozen=True)
class BinaryEvent:
    ticker:     str
    date:       dt.date
    event_type: EventType
    description: str
    # Window (days before/after) to flag as high-volatility
    pre_window:  int = 2
    post_window: int = 3


# ---------------------------------------------------------------------------
# 2024–2025 GLP-1 / Pharma event calendar
# Sources: FDA PDUFA tracker, SEC filings, investor relations pages
# ---------------------------------------------------------------------------

EVENTS: List[BinaryEvent] = [
    # ── Novo Nordisk (NVO) ──────────────────────────────────────────────── #
    BinaryEvent(
        ticker="NVO", date=dt.date(2024, 2, 7),
        event_type=EventType.EARNINGS,
        description="NVO Q4 2023 earnings – Wegovy supply update",
        pre_window=1, post_window=2,
    ),
    BinaryEvent(
        ticker="NVO", date=dt.date(2024, 3, 8),
        event_type=EventType.CLINICAL_TRIAL,
        description="SELECT cardiovascular outcomes trial publication (NEJM)",
        pre_window=3, post_window=5,
    ),
    BinaryEvent(
        ticker="NVO", date=dt.date(2024, 5, 2),
        event_type=EventType.EARNINGS,
        description="NVO Q1 2024 earnings – Ozempic/Wegovy revenue beat",
        pre_window=1, post_window=2,
    ),
    BinaryEvent(
        ticker="NVO", date=dt.date(2024, 6, 21),
        event_type=EventType.FDA_DECISION,
        description="FDA decision on Wegovy cardiovascular indication",
        pre_window=5, post_window=5,
    ),
    BinaryEvent(
        ticker="NVO", date=dt.date(2024, 8, 7),
        event_type=EventType.EARNINGS,
        description="NVO Q2 2024 earnings – GLP-1 demand guidance raised",
        pre_window=1, post_window=2,
    ),
    BinaryEvent(
        ticker="NVO", date=dt.date(2024, 10, 25),
        event_type=EventType.CLINICAL_TRIAL,
        description="FLOW trial renal outcomes readout (EASD Congress)",
        pre_window=3, post_window=3,
    ),
    BinaryEvent(
        ticker="NVO", date=dt.date(2024, 11, 6),
        event_type=EventType.EARNINGS,
        description="NVO Q3 2024 earnings",
        pre_window=1, post_window=2,
    ),
    BinaryEvent(
        ticker="NVO", date=dt.date(2025, 2, 5),
        event_type=EventType.EARNINGS,
        description="NVO Q4 2024 earnings – CagriSema Phase III data miss",
        pre_window=2, post_window=5,
    ),

    # ── Pfizer (PFE) ─────────────────────────────────────────────────────  #
    BinaryEvent(
        ticker="PFE", date=dt.date(2024, 1, 30),
        event_type=EventType.EARNINGS,
        description="PFE Q4 2023 earnings – COVID revenue collapse guidance",
        pre_window=1, post_window=2,
    ),
    BinaryEvent(
        ticker="PFE", date=dt.date(2024, 3, 5),
        event_type=EventType.FDA_DECISION,
        description="FDA decision on Danuglipron (oral GLP-1 candidate)",
        pre_window=3, post_window=3,
    ),
    BinaryEvent(
        ticker="PFE", date=dt.date(2024, 4, 30),
        event_type=EventType.EARNINGS,
        description="PFE Q1 2024 earnings – cost restructuring update",
        pre_window=1, post_window=2,
    ),
    BinaryEvent(
        ticker="PFE", date=dt.date(2024, 7, 30),
        event_type=EventType.EARNINGS,
        description="PFE Q2 2024 earnings",
        pre_window=1, post_window=2,
    ),
    BinaryEvent(
        ticker="PFE", date=dt.date(2024, 10, 15),
        event_type=EventType.CLINICAL_TRIAL,
        description="Danuglipron Phase II dose-finding study withdrawal",
        pre_window=2, post_window=3,
    ),
    BinaryEvent(
        ticker="PFE", date=dt.date(2024, 10, 29),
        event_type=EventType.EARNINGS,
        description="PFE Q3 2024 earnings",
        pre_window=1, post_window=2,
    ),
    BinaryEvent(
        ticker="PFE", date=dt.date(2025, 1, 28),
        event_type=EventType.EARNINGS,
        description="PFE Q4 2024 earnings",
        pre_window=1, post_window=2,
    ),
]


def get_event_dates(
    ticker:      str,
    start:       dt.date,
    end:         dt.date,
    event_types: Sequence[EventType] | None = None,
) -> List[BinaryEvent]:
    """Return all events for `ticker` within [start, end], optionally filtered by type."""
    return [
        e for e in EVENTS
        if e.ticker == ticker
        and start <= e.date <= end
        and (event_types is None or e.event_type in event_types)
    ]


def build_event_mask(
    dates:  "pl.Series",
    ticker: str,
    start:  dt.date,
    end:    dt.date,
) -> "pl.Series":
    """
    Return a boolean Polars Series: True on any date within the pre/post window
    of a binary event for this ticker.

    Used by the Statistics module to isolate event-window returns separately.
    """
    import polars as pl

    events = get_event_dates(ticker, start, end)
    flagged: set[dt.date] = set()
    for ev in events:
        for delta in range(-ev.pre_window, ev.post_window + 1):
            flagged.add(ev.date + dt.timedelta(days=delta))

    flagged_series = pl.Series(sorted(flagged), dtype=pl.Date)
    return dates.is_in(flagged_series)
