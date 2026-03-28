"""
core/types.py — Primitive type aliases and enumerations.

Quant Why: Integer prices (in basis points or cents) eliminate floating-point
comparison bugs that would cause incorrect price-level matching in production.
A $100.125 price stored as 10012500 (in tenths of a cent) avoids the classic
0.1 + 0.2 != 0.3 trap that can silently corrupt order routing logic.
"""

from enum import Enum, auto
from typing import NewType

# --- Scalar primitives ----------------------------------------------------- #

Price = NewType("Price", int)   # e.g. 10012500 = $100.125 in 1/10-cent ticks
Qty   = NewType("Qty",   int)   # always integer lots / shares
OrderId = NewType("OrderId", int)


# --- Enumerations ---------------------------------------------------------- #

class Side(Enum):
    BUY  = auto()
    SELL = auto()

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(Enum):
    LIMIT   = auto()
    MARKET  = auto()
    ICEBERG = auto()   # display_qty visible; rest hidden until refresh


class OrderStatus(Enum):
    OPEN      = auto()
    PARTIAL   = auto()
    FILLED    = auto()
    CANCELLED = auto()


class TimeInForce(Enum):
    GTC = auto()   # Good Till Cancelled
    IOC = auto()   # Immediate Or Cancel
    FOK = auto()   # Fill Or Kill
    DAY = auto()
