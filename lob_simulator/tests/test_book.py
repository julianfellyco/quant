"""
tests/test_book.py — Unit tests for the LOB matching engine.

Coverage targets:
  - Basic limit order resting and matching
  - FIFO queue priority within a price level
  - Full fill and partial fill book cleanup
  - Order cancellation (O(1) splice)
  - Iceberg: display exhaustion → hidden reload → tail re-queue
  - Iceberg: queue position lost after refresh
  - Market order walk-through multiple levels
  - Spread / mid-price calculations
  - Duplicate order_id rejection
"""

import pytest

from lob_simulator.core.book import Book, Fill
from lob_simulator.core.order import Order
from lob_simulator.core.types import OrderId, OrderType, Price, Qty, Side, TimeInForce


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

_oid = 0

def new_oid() -> OrderId:
    global _oid
    _oid += 1
    return OrderId(_oid)


def limit(side: Side, price: int, qty: int, **kwargs) -> Order:
    return Order(
        order_id=new_oid(),
        side=side,
        price=Price(price),
        qty=Qty(qty),
        **kwargs,
    )


def iceberg(side: Side, price: int, display: int, hidden: int) -> Order:
    return Order(
        order_id=new_oid(),
        side=side,
        price=Price(price),
        qty=Qty(display),
        order_type=OrderType.ICEBERG,
        display_qty=Qty(display),
        hidden_qty=Qty(hidden),
    )


def market(side: Side, qty: int) -> Order:
    return Order(
        order_id=new_oid(),
        side=side,
        price=Price(0),
        qty=Qty(qty),
        order_type=OrderType.MARKET,
    )


# --------------------------------------------------------------------------- #
# Basic matching                                                                #
# --------------------------------------------------------------------------- #

class TestBasicMatching:
    def test_no_cross_no_fill(self):
        book = Book()
        b = limit(Side.BUY,  100, 100)
        a = limit(Side.SELL, 101, 100)
        book.add_order(b)
        fills = book.add_order(a)
        assert fills == []
        assert book.best_bid == 100
        assert book.best_ask == 101

    def test_simple_full_cross(self):
        book = Book()
        book.add_order(limit(Side.BUY, 100, 100))
        aggressive = limit(Side.SELL, 100, 100)
        fills = book.add_order(aggressive)

        assert len(fills) == 1
        assert fills[0].qty == 100
        assert fills[0].price == 100
        # Both sides removed
        assert book.best_bid is None
        assert book.best_ask is None

    def test_partial_fill_residual_rests(self):
        book = Book()
        book.add_order(limit(Side.BUY, 100, 200))
        aggressive = limit(Side.SELL, 100, 50)
        fills = book.add_order(aggressive)

        assert len(fills) == 1
        assert fills[0].qty == 50
        # Residual 150 should rest in book
        assert book.best_bid == 100

    def test_aggressive_partial_fill_rests_remainder(self):
        book = Book()
        book.add_order(limit(Side.SELL, 100, 50))
        aggressive = limit(Side.BUY, 100, 200)
        fills = book.add_order(aggressive)

        assert len(fills) == 1
        assert fills[0].qty == 50
        # 150 remains as passive bid
        assert book.best_bid == 100
        assert book.best_ask is None


# --------------------------------------------------------------------------- #
# FIFO queue priority                                                           #
# --------------------------------------------------------------------------- #

class TestQueuePriority:
    def test_fifo_within_level(self):
        """
        Two bids at the same price. When a sell arrives, the earlier bid
        (first enqueued) should be filled first.
        """
        book = Book()
        first  = limit(Side.BUY, 100, 50)
        second = limit(Side.BUY, 100, 50)
        book.add_order(first)
        book.add_order(second)

        fills = book.add_order(limit(Side.SELL, 100, 50))
        assert len(fills) == 1
        assert fills[0].passive_id == first.order_id

    def test_price_priority_beats_time_priority(self):
        """
        Better-priced order fills first even if it arrived later.
        """
        book = Book()
        lower_bid  = limit(Side.BUY, 99,  100)
        higher_bid = limit(Side.BUY, 100, 100)
        book.add_order(lower_bid)
        book.add_order(higher_bid)

        fills = book.add_order(market(Side.SELL, 100))
        assert len(fills) == 1
        assert fills[0].passive_id == higher_bid.order_id
        assert fills[0].price == 100

    def test_qty_ahead_tracking(self):
        book = Book()
        o1 = limit(Side.BUY, 100, 100)
        o2 = limit(Side.BUY, 100, 200)
        book.add_order(o1)
        book.add_order(o2)

        level = book._bids[Price(100)]
        assert level.qty_ahead_of(o1) == 0
        assert level.qty_ahead_of(o2) == 100   # o1 is ahead


# --------------------------------------------------------------------------- #
# Cancellation                                                                  #
# --------------------------------------------------------------------------- #

class TestCancellation:
    def test_cancel_removes_from_book(self):
        book = Book()
        o = limit(Side.BUY, 100, 100)
        book.add_order(o)
        assert book.cancel_order(o.order_id) is True
        assert book.best_bid is None

    def test_cancel_middle_of_queue(self):
        book = Book()
        o1 = limit(Side.BUY, 100, 100)
        o2 = limit(Side.BUY, 100, 200)
        o3 = limit(Side.BUY, 100, 300)
        book.add_order(o1)
        book.add_order(o2)
        book.add_order(o3)

        book.cancel_order(o2.order_id)

        level = book._bids[Price(100)]
        order_ids = [o.order_id for o in level]
        assert o2.order_id not in order_ids
        assert order_ids == [o1.order_id, o3.order_id]
        assert level.total_visible_qty == 400

    def test_cancel_nonexistent_returns_false(self):
        book = Book()
        assert book.cancel_order(OrderId(999)) is False

    def test_cancel_then_fill_does_not_double_remove(self):
        book = Book()
        o = limit(Side.BUY, 100, 100)
        book.add_order(o)
        book.cancel_order(o.order_id)
        # Aggressive sell should find nothing to match
        fills = book.add_order(market(Side.SELL, 100))
        assert fills == []


# --------------------------------------------------------------------------- #
# Iceberg orders                                                                #
# --------------------------------------------------------------------------- #

class TestIceberg:
    def test_iceberg_shows_only_display_qty(self):
        book = Book()
        berg = iceberg(Side.BUY, 100, display=500, hidden=9500)
        book.add_order(berg)

        bids, _ = book.depth(1)
        assert bids[0][1] == 500   # only display portion visible

    def test_iceberg_refresh_after_display_exhausted(self):
        """
        Sell 500 shares against a 500-display / 9500-hidden iceberg.
        After fill, display should reload to 500 and hidden should be 9000.
        """
        book = Book()
        berg = iceberg(Side.BUY, 100, display=500, hidden=9500)
        book.add_order(berg)

        fills = book.add_order(market(Side.SELL, 500))
        assert len(fills) == 1
        assert fills[0].qty == 500

        # Iceberg should still be in book with refreshed display
        bids, _ = book.depth(1)
        assert len(bids) == 1, "Iceberg should still be resting after refresh"
        assert bids[0][1] == 500   # new display slice

        refreshed = book.get_order(berg.order_id)
        assert refreshed is not None
        assert refreshed.display_qty == 500
        assert refreshed.hidden_qty  == 9000
        assert refreshed.leaves_qty  == 9500

    def test_iceberg_queue_position_lost_after_refresh(self):
        """
        Iceberg refreshes go to tail → a new order inserted during the
        refresh interval has priority over the iceberg's refreshed slice.
        """
        book = Book()
        berg = iceberg(Side.BUY, 100, display=100, hidden=400)
        other = limit(Side.BUY, 100, 200)
        book.add_order(berg)
        book.add_order(other)

        # Exhaust berg's display (100 shares) → triggers refresh → goes to tail
        book.add_order(market(Side.SELL, 100))

        level = book._bids[Price(100)]
        # After refresh: order should be [other (200), berg_refreshed (100)]
        order_ids = [o.order_id for o in level]
        assert order_ids.index(other.order_id) < order_ids.index(berg.order_id), \
            "Iceberg should be behind 'other' after refresh"

    def test_iceberg_full_depletion(self):
        """
        Fill an entire 1000-share iceberg (display=200, hidden=800)
        in four 200-share tranches. After depletion it should leave the book.
        """
        book = Book()
        berg = iceberg(Side.BUY, 100, display=200, hidden=800)
        book.add_order(berg)

        for _ in range(5):   # 5 × 200 = 1000 total
            book.add_order(market(Side.SELL, 200))

        assert book.best_bid is None
        assert book.get_order(berg.order_id) is None

    def test_iceberg_partial_hidden_refresh(self):
        """
        If hidden_qty < peak_size, the final refresh only loads what remains.
        """
        book = Book()
        berg = iceberg(Side.BUY, 100, display=300, hidden=100)
        book.add_order(berg)

        # Exhaust display (300)
        book.add_order(market(Side.SELL, 300))
        refreshed = book.get_order(berg.order_id)
        assert refreshed.display_qty == 100   # only 100 left in hidden
        assert refreshed.hidden_qty  == 0


# --------------------------------------------------------------------------- #
# Multi-level market order                                                      #
# --------------------------------------------------------------------------- #

class TestMultiLevel:
    def test_market_order_walks_multiple_levels(self):
        book = Book()
        book.add_order(limit(Side.SELL, 100, 50))
        book.add_order(limit(Side.SELL, 101, 50))
        book.add_order(limit(Side.SELL, 102, 50))

        fills = book.add_order(market(Side.BUY, 130))

        prices = [f.price for f in fills]
        assert Price(100) in prices
        assert Price(101) in prices
        assert Price(102) in prices
        assert sum(f.qty for f in fills) == 130

        # 20 passive shares remain at level 102 (the aggressive market order is gone)
        assert book.best_ask == 102
        _, asks = book.depth(1)
        assert asks[0][1] == 20

    def test_spread_and_mid(self):
        book = Book()
        book.add_order(limit(Side.BUY,  9950, 100))
        book.add_order(limit(Side.SELL, 10050, 100))

        assert book.best_bid  == 9950
        assert book.best_ask  == 10050
        assert book.spread    == 100
        assert book.mid_price == 10000.0


# --------------------------------------------------------------------------- #
# Edge cases                                                                    #
# --------------------------------------------------------------------------- #

class TestEdgeCases:
    def test_duplicate_order_id_raises(self):
        book = Book()
        o = limit(Side.BUY, 100, 100)
        book.add_order(o)
        with pytest.raises(ValueError):
            book.add_order(o)

    def test_ioc_unfilled_residual_cancelled(self):
        book = Book()
        # No asks in book; IOC buy should be cancelled, not rested
        ioc = limit(Side.BUY, 100, 100, tif=TimeInForce.IOC)
        fills = book.add_order(ioc)
        assert fills == []
        assert book.best_bid is None   # not rested

    def test_empty_book_depth(self):
        book = Book()
        bids, asks = book.depth()
        assert bids == []
        assert asks == []
