"""
core/level.py — LimitLevel: a price bucket implemented as a doubly-linked list.

Quant Why: The canonical LOB data structure is a Map<Price, DoublyLinkedList>.
  - Insertion (new order)  → O(1)  append to tail
  - Removal (cancel/fill)  → O(1)  pointer splice (no traversal needed)
  - Best-price access      → O(1)  via the sorted outer map (see book.py)

Compare to a simple Python list where cancellation is O(n) due to index
search. At 50k order updates/sec the difference is measurable.

The level also maintains a running `total_visible_qty` so that depth-of-book
queries (needed for slippage & fill-probability calculations) are O(1) reads
rather than O(n) summations.
"""

from __future__ import annotations

from typing import Iterator, Optional

from .order import Order
from .types import OrderId, Price, Qty


class LimitLevel:
    """
    A single price bucket in the order book.

    Invariants maintained at all times:
      - head.prev is None
      - tail.next is None
      - total_visible_qty == sum(o.visible_qty for o in self)
      - order_count       == len(list(self))
    """

    __slots__ = ("price", "total_visible_qty", "order_count", "head", "tail")

    def __init__(self, price: Price) -> None:
        self.price:             Price          = price
        self.total_visible_qty: Qty            = Qty(0)
        self.order_count:       int            = 0
        self.head:              Optional[Order] = None
        self.tail:              Optional[Order] = None

    # ------------------------------------------------------------------ #
    # Mutations                                                             #
    # ------------------------------------------------------------------ #

    def add_order(self, order: Order) -> None:
        """
        Enqueue at the tail — O(1).

        All new orders, including iceberg refreshes, are appended here.
        Refreshes therefore *lose* their prior queue position, which is the
        correct exchange semantics (CME, NASDAQ, LSE all behave this way).
        """
        if self.tail is None:
            # List is empty
            self.head = order
            self.tail = order
            order.prev = None
            order.next = None
        else:
            order.prev      = self.tail
            order.next      = None
            self.tail.next  = order
            self.tail       = order

        self.total_visible_qty = Qty(self.total_visible_qty + order.visible_qty)
        self.order_count      += 1

    def remove_order(self, order: Order) -> None:
        """
        Splice out an arbitrary node — O(1).

        Used for explicit cancellations and full fills.  Partial fills that
        leave the order live do NOT call remove; instead the book adjusts
        total_visible_qty in-place via `deduct_visible`.
        """
        # Rewire neighbours
        if order.prev is not None:
            order.prev.next = order.next
        else:
            self.head = order.next   # order was head

        if order.next is not None:
            order.next.prev = order.prev
        else:
            self.tail = order.prev   # order was tail

        # Detach pointers so the Order object can be GC'd / reused safely
        order.prev = None
        order.next = None

        self.total_visible_qty = Qty(self.total_visible_qty - order.visible_qty)
        self.order_count      -= 1

    def deduct_visible(self, qty: Qty) -> None:
        """
        Adjust running depth counter after a *partial* fill of the head order.
        Called by the matching engine without removing the order from the list.
        """
        self.total_visible_qty = Qty(self.total_visible_qty - qty)

    # ------------------------------------------------------------------ #
    # Accessors                                                             #
    # ------------------------------------------------------------------ #

    @property
    def is_empty(self) -> bool:
        return self.head is None

    def qty_ahead_of(self, order: Order) -> Qty:
        """
        Compute how many visible shares sit in front of `order` in this level.

        Quant Why: A passive limit order's expected fill time is driven by the
        volume that must trade *through* the queue before it gets hit.  This
        figure feeds the fill-probability model in metrics/execution.py.

        Complexity: O(k) where k = number of orders ahead.  Acceptable because
        this is called only for analytics, never in the hot matching path.
        """
        ahead: Qty = Qty(0)
        cursor = self.head
        while cursor is not None and cursor is not order:
            ahead = Qty(ahead + cursor.visible_qty)
            cursor = cursor.next
        return ahead

    # ------------------------------------------------------------------ #
    # Iteration                                                             #
    # ------------------------------------------------------------------ #

    def __iter__(self) -> Iterator[Order]:
        cursor = self.head
        while cursor is not None:
            yield cursor
            cursor = cursor.next

    def __len__(self) -> int:
        return self.order_count

    def __repr__(self) -> str:
        return (
            f"LimitLevel(price={self.price}, "
            f"visible_qty={self.total_visible_qty}, "
            f"orders={self.order_count})"
        )
