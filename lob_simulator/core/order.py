"""
core/order.py — Order node definition.

The Order object serves a dual role:
  1. A value object carrying business data (price, qty, side, type).
  2. A node in the doubly-linked list inside a LimitLevel.

Quant Why: Embedding prev/next pointers directly in the Order object (rather
than wrapping it in a separate node class) avoids one heap allocation and one
pointer indirection per order — material at 100k+ orders/sec in a real gateway.
This is the same pattern used in Linux kernel list.h.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .types import OrderId, OrderStatus, OrderType, Price, Qty, Side, TimeInForce


@dataclass
class Order:
    # ------------------------------------------------------------------ #
    # Identity & routing                                                    #
    # ------------------------------------------------------------------ #
    order_id:     OrderId
    side:         Side
    price:        Price        # 0 for MARKET orders
    qty:          Qty          # *remaining* displayable quantity
    order_type:   OrderType    = OrderType.LIMIT
    tif:          TimeInForce  = TimeInForce.GTC

    # ------------------------------------------------------------------ #
    # Iceberg fields                                                         #
    # ------------------------------------------------------------------ #
    # For a 10,000-share iceberg showing 500 at a time:
    #   display_qty = 500   (what the book shows; decrements on fills)
    #   hidden_qty  = 9500  (reservoir; decrements on each refresh)
    #   _peak_size  = 500   (original display_qty, used to reload)
    display_qty: Qty = field(default=0)
    hidden_qty:  Qty = field(default=0)
    _peak_size:  Qty = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Execution state                                                        #
    # ------------------------------------------------------------------ #
    status:          OrderStatus = field(default=OrderStatus.OPEN)
    filled_qty:      Qty         = field(default=Qty(0))
    leaves_qty:      Qty         = field(default=Qty(0), init=False)  # set in __post_init__
    avg_fill_price:  float       = field(default=0.0)
    _fill_notional:  float       = field(default=0.0, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Timing & queue-position tracking                                      #
    # ------------------------------------------------------------------ #
    timestamp:      float = field(default_factory=time.monotonic)
    # qty_ahead tracks shares in front of this order at the same price level.
    # Updated lazily on each book snapshot — used for fill-probability models.
    qty_ahead:      Qty   = field(default=Qty(0), init=False)

    # ------------------------------------------------------------------ #
    # Doubly-linked list pointers (owned by LimitLevel)                    #
    # ------------------------------------------------------------------ #
    prev: Optional[Order] = field(default=None, init=False, repr=False)
    next: Optional[Order] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        if self.order_type is OrderType.ICEBERG:
            if self.display_qty <= 0:
                raise ValueError("Iceberg order requires display_qty > 0")
            if self.hidden_qty < 0:
                raise ValueError("hidden_qty must be >= 0")
            self._peak_size = self.display_qty
            self.leaves_qty = Qty(self.display_qty + self.hidden_qty)
        else:
            self.display_qty = self.qty
            self.leaves_qty  = self.qty

    # ------------------------------------------------------------------ #
    # Properties                                                            #
    # ------------------------------------------------------------------ #

    @property
    def is_iceberg(self) -> bool:
        return self.order_type is OrderType.ICEBERG

    @property
    def visible_qty(self) -> Qty:
        """Quantity visible to the rest of the market (book depth)."""
        return self.display_qty if self.is_iceberg else self.qty

    @property
    def total_remaining(self) -> Qty:
        """Unfilled quantity across display + hidden tiers."""
        return self.leaves_qty

    @property
    def is_live(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)

    # ------------------------------------------------------------------ #
    # Execution helpers                                                     #
    # ------------------------------------------------------------------ #

    def apply_fill(self, fill_qty: Qty, fill_price: Price) -> None:
        """
        Record a (partial) fill against this order.
        Updates running average fill price via incremental formula to avoid
        accumulating floating-point error over many partial fills.
        """
        if fill_qty <= 0:
            raise ValueError("fill_qty must be positive")

        # Update VWAP fill price incrementally
        self._fill_notional += fill_qty * fill_price
        self.filled_qty     = Qty(self.filled_qty + fill_qty)
        self.avg_fill_price  = self._fill_notional / self.filled_qty

        # Decrement the visible tier first
        if self.is_iceberg:
            self.display_qty = Qty(self.display_qty - fill_qty)
        else:
            self.qty       = Qty(self.qty - fill_qty)
            self.display_qty = self.qty

        self.leaves_qty = Qty(self.leaves_qty - fill_qty)

        if self.leaves_qty == 0:
            self.status = OrderStatus.FILLED
        elif self.filled_qty > 0:
            self.status = OrderStatus.PARTIAL

    def needs_refresh(self) -> bool:
        """True when the display slice is exhausted and hidden supply remains."""
        return self.is_iceberg and self.display_qty == 0 and self.hidden_qty > 0

    def refresh_display(self) -> None:
        """
        Reload display_qty from the hidden reservoir.

        Quant Why: After refresh the order loses its queue priority and is
        re-inserted at the *tail* of its price level — equivalent to a new
        order arriving at that price. This is intentional: exchanges penalise
        iceberg refreshes to discourage gaming of displayed depth, ensuring
        that orders which reveal size early are rewarded with better queue
        position.
        """
        if not self.needs_refresh():
            raise RuntimeError("refresh_display called on non-exhausted iceberg")

        refill = min(self._peak_size, self.hidden_qty)
        self.display_qty = Qty(refill)
        self.hidden_qty  = Qty(self.hidden_qty - refill)
        # leaves_qty is unchanged (already tracked correctly)
