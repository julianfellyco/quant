"""
metrics/ofi.py — Order Flow Imbalance (OFI) tracker.

What is OFI?
------------
Order Flow Imbalance measures the net directional pressure of limit orders
at the Best Bid and Offer (BBO).  Between two book snapshots:

    ΔBid = BestBidQty_t − BestBidQty_{t−1}  (+ if depth grew, − if shrank)
    ΔAsk = BestAskQty_t − BestAskQty_{t−1}
    OFI  = ΔBid − ΔAsk

A positive OFI means buy-side depth built up faster than sell-side depth,
indicating net buy pressure — a leading indicator of upward price movement.

Normalised OFI ∈ [−1, 1]:
    NOFI = OFI / (|ΔBid| + |ΔAsk| + ε)

Quant Why (Cont, Kukanov, Stoikov 2014)
---------------------------------------
OFI has significantly higher predictive power for short-term mid-price
changes than the signed trade imbalance (which only looks at executed trades).
The intuition: a large build-up of buy-side resting limit orders reflects
informed expectations about near-term price moves, even before those orders
are executed.

In a high-frequency context, a 10-second cumulative OFI is predictive of
the 1-minute price move with R² often > 0.6 on liquid equity futures.

Usage
-----
>>> from lob_simulator.core.book import Book
>>> from lob_simulator.metrics.ofi import OFITracker
>>>
>>> book = Book()
>>> tracker = OFITracker(book)
>>> tracker.snapshot()              # capture initial state
>>> book.add_order(buy_order)
>>> ofi = tracker.snapshot()        # OFI for this interval
>>> cum = tracker.cumulative_ofi(window=20)
"""

from __future__ import annotations

from typing import Optional

from ..core.book import Book
from ..core.types import Qty


class OFITracker:
    """
    Snapshot-based Order Flow Imbalance tracker.

    Attach one tracker to a Book instance.  Call snapshot() after each
    meaningful event (or at each bar boundary) to record the OFI for that
    interval.  The tracker stores raw OFI values internally; use the helper
    methods to compute rolling statistics.

    Parameters
    ----------
    book : Book
        The LOB instance to observe.
    epsilon : float
        Small constant in the NOFI denominator to avoid division by zero.
        Default 1.0 (one share), which is negligible for typical lot sizes.
    """

    def __init__(self, book: Book, epsilon: float = 1.0) -> None:
        self._book      = book
        self._epsilon   = epsilon

        # Previous snapshot state
        self._prev_bid_qty: Optional[int] = None
        self._prev_ask_qty: Optional[int] = None

        # History: raw OFI per snapshot interval
        self._ofi_history:  list[float] = []
        # History: normalised OFI ∈ [−1, 1]
        self._nofi_history: list[float] = []

    # ------------------------------------------------------------------ #
    # Core snapshot                                                         #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> Optional[float]:
        """
        Record the current BBO quantities and return the OFI for this interval.

        Returns None on the first call (no previous state to diff against).
        Subsequent calls return a float.

        OFI sign convention:
            > 0 : buy-side pressure (bid depth grew more than ask depth)
            < 0 : sell-side pressure
            = 0 : balanced or no change on either side
        """
        bids, asks = self._book.depth(levels=1)

        cur_bid_qty: int = bids[0][1] if bids else 0
        cur_ask_qty: int = asks[0][1] if asks else 0

        if self._prev_bid_qty is None:
            # First snapshot — record state, return None (no interval yet)
            self._prev_bid_qty = cur_bid_qty
            self._prev_ask_qty = cur_ask_qty
            return None

        delta_bid = cur_bid_qty - self._prev_bid_qty
        delta_ask = cur_ask_qty - (self._prev_ask_qty or 0)

        raw_ofi  = float(delta_bid - delta_ask)
        denom    = abs(delta_bid) + abs(delta_ask) + self._epsilon
        norm_ofi = raw_ofi / denom

        self._ofi_history.append(raw_ofi)
        self._nofi_history.append(norm_ofi)

        # Advance state
        self._prev_bid_qty = cur_bid_qty
        self._prev_ask_qty = cur_ask_qty

        return raw_ofi

    # ------------------------------------------------------------------ #
    # Rolling statistics                                                    #
    # ------------------------------------------------------------------ #

    def cumulative_ofi(self, window: int = 10) -> float:
        """
        Rolling sum of raw OFI over the last `window` snapshots.

        A large positive value indicates sustained buy-side pressure over the
        window — the primary signal for short-term directional prediction.

        Returns 0.0 if fewer than `window` snapshots have been recorded.
        """
        if len(self._ofi_history) < window:
            return 0.0
        return sum(self._ofi_history[-window:])

    def normalised_ofi(self, window: int = 10) -> float:
        """
        Rolling mean of normalised OFI (NOFI) over the last `window` snapshots.

        NOFI ∈ [−1, 1].  A value of +0.8 means that 80% of the directional
        pressure in the window was on the buy side.

        Returns 0.0 if fewer than `window` snapshots have been recorded.
        """
        if len(self._nofi_history) < window:
            return 0.0
        return sum(self._nofi_history[-window:]) / window

    def last_ofi(self) -> float:
        """Most recent raw OFI value, or 0.0 if no snapshots yet."""
        return self._ofi_history[-1] if self._ofi_history else 0.0

    def last_nofi(self) -> float:
        """Most recent normalised OFI, or 0.0 if no snapshots yet."""
        return self._nofi_history[-1] if self._nofi_history else 0.0

    # ------------------------------------------------------------------ #
    # Full history                                                          #
    # ------------------------------------------------------------------ #

    @property
    def ofi_series(self) -> list[float]:
        """All raw OFI values recorded so far (copy)."""
        return list(self._ofi_history)

    @property
    def nofi_series(self) -> list[float]:
        """All normalised OFI values recorded so far (copy)."""
        return list(self._nofi_history)

    def reset(self) -> None:
        """Clear history and reset previous-state memory."""
        self._prev_bid_qty  = None
        self._prev_ask_qty  = None
        self._ofi_history.clear()
        self._nofi_history.clear()

    def __repr__(self) -> str:
        return (
            f"OFITracker("
            f"snapshots={len(self._ofi_history)}, "
            f"last_ofi={self.last_ofi():.1f}, "
            f"last_nofi={self.last_nofi():.3f})"
        )
