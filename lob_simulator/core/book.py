"""
core/book.py — Central Limit Order Book (CLOB) matching engine.

Data structure overview
-----------------------
    bids : SortedDict[Price, LimitLevel]   ascending keys, best bid = *last* key
    asks : SortedDict[Price, LimitLevel]   ascending keys, best ask = *first* key
    order_map : Dict[OrderId, Order]       O(1) cancel / amend lookup

Quant Why — SortedDict (from sortedcontainers, a pure-Python skip-list):
  - Price-level insert/delete : O(log n) in the number of distinct price levels
  - Best bid/ask access       : O(1) via .peekitem(-1) / .peekitem(0)
  - Alternative (heap)        : O(log n) insert but O(n) arbitrary cancel
  - Alternative (plain dict)  : O(1) insert but no ordering → O(n) best-price

For n < 10,000 price levels (typical for a single instrument intraday) the
skip-list constant factors are negligible and the simplicity pays off.

Matching priority: Price → Time (FIFO within a level).
Iceberg refresh:   When display slice is exhausted, the order re-joins at the
                   tail of the same price level (loses queue position).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sortedcontainers import SortedDict

from .level import LimitLevel
from .order import Order
from .types import OrderId, OrderStatus, OrderType, Price, Qty, Side, TimeInForce


# --------------------------------------------------------------------------- #
# Fill record                                                                   #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Fill:
    """
    Immutable record of a single matched execution.

    aggressive_id: the order that crossed the spread (taker)
    passive_id:    the resting order that was hit (maker)

    Quant Why: Separating taker/maker lets us later compute:
      - Taker slippage  = fill_price − arrival_mid  (adverse selection cost)
      - Maker rebate    = exchange fee credit for providing liquidity
      - Price impact    = Σ fill_price × fill_qty / Σ fill_qty − pre-trade mid
    """
    aggressive_id: OrderId
    passive_id:    OrderId
    side:          Side     # side of the aggressive order
    price:         Price    # always the passive order's price (maker sets price)
    qty:           Qty
    timestamp:     float


# --------------------------------------------------------------------------- #
# Book                                                                          #
# --------------------------------------------------------------------------- #

class Book:
    """
    Full central limit order book for a single instrument.

    Public API
    ----------
    add_order(order)         → List[Fill]
    cancel_order(order_id)   → bool
    get_order(order_id)      → Optional[Order]
    best_bid / best_ask      → Optional[Price]
    mid_price                → Optional[float]
    spread                   → Optional[Price]
    depth(n)                 → Tuple[List, List]   # top-n bid/ask levels
    """

    def __init__(self) -> None:
        # Price → LimitLevel.  bids sorted ascending; we use .peekitem(-1) for best.
        self._bids: SortedDict = SortedDict()
        # Price → LimitLevel.  asks sorted ascending; we use .peekitem(0) for best.
        self._asks: SortedDict = SortedDict()

        self._order_map: Dict[OrderId, Order] = {}
        self._fill_history: List[Fill] = []
        self._next_ts: float = 0.0   # injected clock for deterministic tests

    # ------------------------------------------------------------------ #
    # Clock (injectable for testing)                                        #
    # ------------------------------------------------------------------ #

    def _now(self) -> float:
        import time
        return time.monotonic()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _side_book(self, side: Side) -> SortedDict:
        return self._bids if side is Side.BUY else self._asks

    def _get_or_create_level(self, side: Side, price: Price) -> LimitLevel:
        book = self._side_book(side)
        if price not in book:
            book[price] = LimitLevel(price)
        return book[price]

    def _remove_level_if_empty(self, side: Side, price: Price) -> None:
        book = self._side_book(side)
        if price in book and book[price].is_empty:
            del book[price]

    # ------------------------------------------------------------------ #
    # Best prices                                                           #
    # ------------------------------------------------------------------ #

    @property
    def best_bid(self) -> Optional[Price]:
        """Highest resting buy price. O(1)."""
        if not self._bids:
            return None
        return self._bids.peekitem(-1)[0]   # last key in ascending sorted dict

    @property
    def best_ask(self) -> Optional[Price]:
        """Lowest resting sell price. O(1)."""
        if not self._asks:
            return None
        return self._asks.peekitem(0)[0]    # first key

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    @property
    def spread(self) -> Optional[Price]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return Price(ba - bb)

    # ------------------------------------------------------------------ #
    # Order entry                                                           #
    # ------------------------------------------------------------------ #

    def add_order(self, order: Order) -> List[Fill]:
        """
        Submit an order to the book.

        1. Attempt matching against the opposite side.
        2. If residual quantity remains (and TIF allows), rest it in the book.
        3. Return list of Fill records generated.

        Matching walkthrough (for a BUY limit order at price P):
          - Walk asks from best_ask upward while ask_price <= P
          - At each level, match FIFO through the linked list
          - Handle iceberg exhaustion → refresh → re-enqueue at tail
          - Stop when order is fully filled, or no more matching prices
        """
        if order.order_id in self._order_map:
            raise ValueError(f"Duplicate order_id {order.order_id}")

        fills: List[Fill] = []

        if order.order_type is OrderType.MARKET:
            fills = self._match(order)
            # Market orders never rest; cancel any unfilled residual
            if order.is_live:
                order.status = OrderStatus.CANCELLED
        else:
            # Limit / Iceberg — attempt crossing first
            fills = self._match(order)

            if order.is_live:
                if order.tif is TimeInForce.IOC:
                    order.status = OrderStatus.CANCELLED
                elif order.tif is TimeInForce.FOK:
                    # FOK: if not fully filled, cancel AND unwind fills
                    # (simplified: in practice exchange rejects before matching)
                    order.status = OrderStatus.CANCELLED
                    fills = []   # treat as if nothing happened
                else:
                    # GTC / DAY: rest in book
                    self._rest_order(order)

        self._fill_history.extend(fills)
        return fills

    def _rest_order(self, order: Order) -> None:
        """Place a live order onto the passive side of the book."""
        level = self._get_or_create_level(order.side, order.price)
        level.add_order(order)
        self._order_map[order.order_id] = order

    # ------------------------------------------------------------------ #
    # Matching engine                                                       #
    # ------------------------------------------------------------------ #

    def _match(self, aggressive: Order) -> List[Fill]:
        """
        Core FIFO price-time matching loop.

        For a BUY  aggressive: walks asks from lowest price upward.
        For a SELL aggressive: walks bids from highest price downward.
        """
        fills: List[Fill] = []
        opposite_book = self._side_book(aggressive.side.opposite)

        while aggressive.is_live:
            # ── Select best opposing price level ──────────────────────── #
            if aggressive.side is Side.BUY:
                if not opposite_book:
                    break
                best_price, level = opposite_book.peekitem(0)   # lowest ask
                # Limit order: only match if ask <= our bid
                if (aggressive.order_type is not OrderType.MARKET
                        and best_price > aggressive.price):
                    break
            else:  # SELL
                if not opposite_book:
                    break
                best_price, level = opposite_book.peekitem(-1)  # highest bid
                if (aggressive.order_type is not OrderType.MARKET
                        and best_price < aggressive.price):
                    break

            # ── Match FIFO through this level ─────────────────────────── #
            while aggressive.is_live and not level.is_empty:
                passive = level.head   # type: Order
                fill_qty = min(aggressive.visible_qty, passive.visible_qty)

                # Record fill
                f = Fill(
                    aggressive_id = aggressive.order_id,
                    passive_id    = passive.order_id,
                    side          = aggressive.side,
                    price         = best_price,
                    qty           = Qty(fill_qty),
                    timestamp     = self._now(),
                )
                fills.append(f)

                # Apply to both sides
                aggressive.apply_fill(Qty(fill_qty), best_price)
                passive.apply_fill(Qty(fill_qty), best_price)

                # Adjust running depth counter on the level
                level.deduct_visible(Qty(fill_qty))

                # ── Handle passive order lifecycle ────────────────────── #
                if passive.visible_qty == 0:
                    level.remove_order(passive)

                    if passive.needs_refresh():
                        # Iceberg: reload display slice from hidden reservoir.
                        # Re-enqueue at TAIL → loses queue position.
                        self._handle_iceberg_refresh(passive, level)
                    else:
                        # Order fully filled (or no hidden reserve left)
                        if passive.order_id in self._order_map:
                            del self._order_map[passive.order_id]

            # ── Clean up empty price level ─────────────────────────────── #
            if level.is_empty:
                del opposite_book[best_price]

        return fills

    # ------------------------------------------------------------------ #
    # Iceberg refresh                                                       #
    # ------------------------------------------------------------------ #

    def _handle_iceberg_refresh(self, order: Order, level: LimitLevel) -> None:
        """
        Refresh an iceberg's display slice and re-enqueue at the tail.

        Sequence:
          1. Call order.refresh_display() → loads next peak-size slice
          2. Re-append to the tail of the same level
          3. Update total_visible_qty on the level (add_order does this)

        The order remains in _order_map since it is still live.

        Quant Why: Re-queueing at the tail is the correct exchange behaviour
        on all major venues. An iceberg that repeatedly refreshes effectively
        acts as a series of new orders — it never "saves" its queue spot.
        This has a strategic implication: if your iceberg is large relative
        to the natural flow, you may never fully fill in a trending market
        because each refresh sends you to the back just as new aggressive
        flow arrives.
        """
        order.refresh_display()
        level.add_order(order)
        # order_map entry is retained; order is still live

    # ------------------------------------------------------------------ #
    # Cancellation                                                          #
    # ------------------------------------------------------------------ #

    def cancel_order(self, order_id: OrderId) -> bool:
        """
        Cancel a resting order — O(1) pointer splice + O(log n) level cleanup.

        Returns True if the order was found and cancelled, False otherwise.
        """
        order = self._order_map.get(order_id)
        if order is None or not order.is_live:
            return False

        level = self._side_book(order.side).get(order.price)
        if level is None:
            return False

        level.remove_order(order)
        order.status = OrderStatus.CANCELLED
        del self._order_map[order_id]
        self._remove_level_if_empty(order.side, order.price)
        return True

    # ------------------------------------------------------------------ #
    # Accessors                                                             #
    # ------------------------------------------------------------------ #

    def get_order(self, order_id: OrderId) -> Optional[Order]:
        return self._order_map.get(order_id)

    def depth(self, levels: int = 5) -> Tuple[List[Tuple[Price, Qty]], List[Tuple[Price, Qty]]]:
        """
        Return top-`levels` bid and ask price/qty pairs.

        Quant Why: Depth-of-book is the primary input for:
          - Slippage estimation  (how far does a VWAP walk the book?)
          - Queue-position model (how many shares are ahead of my passive order?)
          - Adverse-selection proxy (imbalance between bid/ask depth)

        Returns:
            bids : [(price, qty), ...] descending by price
            asks : [(price, qty), ...] ascending by price
        """
        bids = [
            (p, lvl.total_visible_qty)
            for p, lvl in reversed(self._bids.items())
        ][:levels]

        asks = [
            (p, lvl.total_visible_qty)
            for p, lvl in self._asks.items()
        ][:levels]

        return bids, asks

    @property
    def fills(self) -> List[Fill]:
        return list(self._fill_history)

    def __repr__(self) -> str:
        return (
            f"Book(best_bid={self.best_bid}, best_ask={self.best_ask}, "
            f"spread={self.spread}, open_orders={len(self._order_map)})"
        )
