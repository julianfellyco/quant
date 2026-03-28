#!/usr/bin/env python3
"""
scripts/benchmark_latency.py — Per-operation latency benchmark for the
                                LOB matching engine.

Why this benchmark exists
--------------------------
The matching engine makes two O(1) guarantees that must hold at runtime:
  1. Order insertion    → SortedDict price-level lookup + linked-list append
  2. Order cancellation → hash-map lookup + pointer splice (no traversal)
  3. Order matching     → price-level iteration + fill dispatch

A regression in any of these (e.g. accidentally quadratic LimitLevel.add_order)
would surface as a P99 spike long before a full backtest would reveal it.

Threshold: P99 < 1,000 μs (1 ms) per operation.

Exit codes
----------
  0 — all operations within threshold
  1 — one or more operations exceeded threshold

Warm-up rationale
-----------------
CPython uses specialised bytecode caches and per-attribute version tags that
are lazily populated.  The first N calls to a function are always slower than
steady-state.  Running N_WARMUP iterations before measurement eliminates this
source of noise and gives a realistic picture of sustained throughput latency.
"""

from __future__ import annotations

import statistics
import sys
import time

from lob_simulator.core.book import Book
from lob_simulator.core.order import Order
from lob_simulator.core.types import OrderId, OrderType, Price, Qty, Side

# ─────────────────────────────────────────────────────────────────────────── #
# Configuration                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

N_WARMUP          = 1_000       # iterations discarded before measurement begins
N_SAMPLE          = 10_000      # measured iterations per benchmark
THRESHOLD_P99_US  = 1_000.0     # 1 ms expressed in microseconds
WIDTH             = 78


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def _make_limit(oid: int, side: Side, price: int, qty: int = 100) -> Order:
    return Order(
        order_id = OrderId(oid),
        side     = side,
        price    = Price(price),
        qty      = Qty(qty),
    )


def _make_market(oid: int, side: Side, qty: int = 100) -> Order:
    return Order(
        order_id   = OrderId(oid),
        side       = side,
        price      = Price(0),
        qty        = Qty(qty),
        order_type = OrderType.MARKET,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Benchmark functions — each returns a list of per-operation latencies in μs  #
# ─────────────────────────────────────────────────────────────────────────── #

def bench_insert(n: int) -> list[float]:
    """
    Resting limit order insertion with no opposing side.

    Hot path: SortedDict.__setitem__ (O(log P)) + LimitLevel.add_order (O(1)).
    P = number of distinct price levels; kept small (~200) to isolate the
    linked-list cost from SortedDict re-balancing variance.
    """
    book = Book()
    lats: list[float] = []

    for i in range(1, n + 1):
        order = _make_limit(i, Side.BUY, price=10_000 - (i % 200))
        t0 = time.perf_counter_ns()
        book.add_order(order)
        lats.append((time.perf_counter_ns() - t0) / 1_000)

    return lats


def bench_cancel(n: int) -> list[float]:
    """
    O(1) cancellation via order_map lookup + LimitLevel.remove_order splice.

    We pre-populate the book with n resting orders so cancellation does not
    include any insertion time.  Orders are cancelled in FIFO order (same
    sequence they were inserted) so the head-node case is exercised.
    """
    book   = Book()
    orders = [_make_limit(i, Side.BUY, price=10_000) for i in range(1, n + 1)]
    for o in orders:
        book.add_order(o)

    lats: list[float] = []
    for o in orders:
        t0 = time.perf_counter_ns()
        book.cancel_order(o.order_id)
        lats.append((time.perf_counter_ns() - t0) / 1_000)

    return lats


def bench_match(n: int) -> list[float]:
    """
    Single-fill market order match against one resting ask.

    Each iteration constructs a fresh book so we measure a cold match
    (worst case: level lookup + one fill + level cleanup) rather than
    amortising book-teardown cost across iterations.

    Quant Why: in live trading, the first fill on a new price level is the
    critical path.  Deep sweeps are rare; single-fill latency is what matters
    for limit order placement decisions.
    """
    lats: list[float] = []

    for i in range(1, n + 1):
        book = Book()
        book.add_order(_make_limit(i * 2, Side.SELL, price=10_050, qty=200))
        buyer = _make_market(i * 2 + 1, Side.BUY, qty=100)

        t0 = time.perf_counter_ns()
        book.add_order(buyer)
        lats.append((time.perf_counter_ns() - t0) / 1_000)

    return lats


# ─────────────────────────────────────────────────────────────────────────── #
# Statistics (no numpy/scipy — keep the benchmark dependency-free)            #
# ─────────────────────────────────────────────────────────────────────────── #

def percentile(data: list[float], p: float) -> float:
    """
    Linear-interpolation percentile (equivalent to numpy.percentile default).
    Avoids numpy so the benchmark only depends on lob_simulator.
    """
    s  = sorted(data)
    k  = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def print_row(name: str, lats: list[float]) -> bool:
    """
    Print a one-line stats summary and return True iff P99 < THRESHOLD_P99_US.

    Columns: operation name, mean, P50, P99, P99.9, pass/fail verdict.
    """
    mean = statistics.mean(lats)
    p50  = percentile(lats, 50)
    p99  = percentile(lats, 99)
    p999 = percentile(lats, 99.9)
    ok   = p99 < THRESHOLD_P99_US

    verdict = "PASS ✓" if ok else "FAIL ✗"
    print(
        f"  {name:<26}"
        f"  mean={mean:7.2f} μs"
        f"  P50={p50:7.2f} μs"
        f"  P99={p99:7.2f} μs"
        f"  P99.9={p999:8.2f} μs"
        f"  {verdict}"
    )
    if not ok:
        print(
            f"    → P99 {p99:.1f} μs exceeds the "
            f"{THRESHOLD_P99_US:.0f} μs threshold."
        )
    return ok


# ─────────────────────────────────────────────────────────────────────────── #
# Entry point                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def main() -> int:
    bar = "─" * WIDTH
    print(f"\n{'LOB Matching Engine — Latency Benchmark':^{WIDTH}}")
    print(bar)
    print(f"  Samples per operation : {N_SAMPLE:,}")
    print(f"  P99 threshold         : {THRESHOLD_P99_US:,.0f} μs  (= 1 ms)")
    print(f"  Timing resolution     : time.perf_counter_ns()  (~1 ns)")
    print(bar)

    # ── Warm up ──────────────────────────────────────────────────────────── #
    print(f"\n  Warming up ({N_WARMUP:,} iterations per operation)...", end="", flush=True)
    bench_insert(N_WARMUP)
    bench_cancel(N_WARMUP)
    bench_match(N_WARMUP)
    print("  done\n")

    # ── Measure ──────────────────────────────────────────────────────────── #
    suite = [
        ("insert  (no match)",      bench_insert),
        ("cancel  (O(1) splice)",   bench_cancel),
        ("match   (market order)",  bench_match),
    ]

    print(
        f"  {'Operation':<26}"
        f"  {'Mean':>12}"
        f"  {'P50':>12}"
        f"  {'P99':>12}"
        f"  {'P99.9':>14}"
        f"  {'Result'}"
    )
    print(bar)

    results = [(name, fn(N_SAMPLE)) for name, fn in suite]
    all_ok  = all(print_row(name, lats) for name, lats in results)

    # ── Summary ───────────────────────────────────────────────────────────── #
    print(bar)
    if all_ok:
        print(f"\n  All operations within {THRESHOLD_P99_US:.0f} μs P99.  [ PASS ]\n")
        return 0
    else:
        print(f"\n  One or more operations exceeded {THRESHOLD_P99_US:.0f} μs P99.  [ FAIL ]\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
