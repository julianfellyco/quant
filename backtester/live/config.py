"""Live trading configuration and safety limits."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
import datetime as dt


@dataclass
class LiveConfig:
    """Configuration and hard safety limits for paper trading."""

    # Alpaca credentials (paper)
    api_key: str = field(default_factory=lambda: os.environ.get("ALPACA_API_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.environ.get("ALPACA_SECRET_KEY", ""))
    base_url: str = "https://paper-api.alpaca.markets"   # paper only — never change

    # Safety limits (hard-coded — DO NOT make configurable)
    max_order_shares: int = 100          # max shares per order
    max_order_notional: float = 10_000.0 # max notional per order (USD)
    max_daily_orders: int = 20           # circuit breaker: halt after N orders/day
    paper_mode: bool = True              # ALWAYS True — live trading not supported

    # Trading hours (US ET)
    market_open: dt.time = field(default_factory=lambda: dt.time(9, 30))
    market_close: dt.time = field(default_factory=lambda: dt.time(16, 0))
    timezone: str = "America/New_York"

    def is_market_hours(self) -> bool:
        """Return True if current ET time is within regular trading hours."""
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(self.timezone)
        except ImportError:
            import pytz
            tz = pytz.timezone(self.timezone)
        now = dt.datetime.now(tz).time()
        return self.market_open <= now <= self.market_close
