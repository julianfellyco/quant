"""
metrics/execution.py — Execution quality analytics.

Quant Why: A backtest without execution quality metrics is a lie.  You may
generate alpha in signal space but give it all back (and more) in execution.
These metrics quantify where the money goes.

Definitions
-----------
Slippage (implementation shortfall):
    IS = (execution VWAP − decision price) × side_sign
    Positive IS means the trade cost more than the decision price.
    Split into:
      - Market impact   : permanent price change caused by your trade
      - Timing cost     : price drift between decision and execution
      - Spread cost     : half-spread paid by the taker

Fill probability (passive orders):
    P(fill within horizon T) ≈ f(queue_position, avg_trade_rate, volatility)
    Here we implement the simple roll-model: P = 1 − exp(−λ·leaves_qty)
    where λ = avg_volume_rate / qty_at_level  (Poisson arrival approximation).
"""

from __future__ import annotations

import math
from typing import List, Optional

from ..core.book import Fill
from ..core.types import Price, Qty


def implementation_shortfall(
    fills:          List[Fill],
    decision_price: Price,
    side_sign:      int,          # +1 for buy, −1 for sell
) -> float:
    """
    Compute implementation shortfall in price ticks.

    IS > 0  →  paid more than decision price  (bad)
    IS < 0  →  paid less                      (good / lucky)

    Args:
        fills:          execution fills (taker view)
        decision_price: mid-price at signal time
        side_sign:      +1 for buy order, -1 for sell order
    """
    if not fills:
        return 0.0

    total_qty = sum(f.qty for f in fills)
    if total_qty == 0:
        return 0.0

    vwap = sum(f.price * f.qty for f in fills) / total_qty
    return side_sign * (vwap - decision_price)


def slippage_bps(
    fills:          List[Fill],
    decision_price: Price,
    side_sign:      int,
) -> float:
    """Implementation shortfall expressed in basis points."""
    is_ticks = implementation_shortfall(fills, decision_price, side_sign)
    return (is_ticks / decision_price) * 10_000


def fill_vwap(fills: List[Fill]) -> Optional[float]:
    """Volume-weighted average fill price across all executions."""
    total_qty = sum(f.qty for f in fills)
    if total_qty == 0:
        return None
    return sum(f.price * f.qty for f in fills) / total_qty


def fill_probability_passive(
    qty_ahead:        Qty,
    qty_at_level:     Qty,
    avg_volume_rate:  float,   # shares per unit time at this price
    horizon:          float,   # time units to wait
) -> float:
    """
    Poisson-process approximation of fill probability for a resting order.

    Model: volume arriving at the level follows a Poisson process with rate λ.
    The order fills when cumulative arrivals exceed qty_ahead + own_qty.
    We simplify by asking: what fraction of the level's volume trades in [0, T]?

    P(fill) ≈ 1 − exp(−λ · horizon / (qty_at_level + 1))

    This is intentionally simple — a full implementation would use
    Cont-Stoikov or a survival model calibrated to historical trade data.

    Args:
        qty_ahead:       shares in front of this order at the price level
        qty_at_level:    total visible shares at the level
        avg_volume_rate: average shares/second trading at this price
        horizon:         seconds the order is willing to wait
    """
    if qty_at_level <= 0 or avg_volume_rate <= 0:
        return 0.0

    # Expected time for the queue to clear to our position
    time_to_front = qty_ahead / avg_volume_rate
    if time_to_front >= horizon:
        return 0.0

    # Remaining horizon once we reach the front
    residual = horizon - time_to_front
    lam = avg_volume_rate / max(qty_at_level, 1)
    return 1.0 - math.exp(-lam * residual)


def market_impact_estimate(
    order_qty:       Qty,
    adv:             float,   # average daily volume (same units as order_qty)
    volatility_daily: float,  # daily volatility in price-tick units
    eta:             float = 0.1,   # market-impact coefficient (calibrate per asset)
    beta:            float = 0.5,   # square-root law exponent
) -> float:
    """
    Square-root market impact model (Almgren-Chriss family).

    impact = η · σ · (Q / ADV)^β

    Quant Why: Empirically, market impact scales with the square root of
    participation rate (not linearly). Doubling order size does NOT double
    impact — the exponent β ≈ 0.5 is remarkably universal across asset classes.
    This formula is the foundation of all institutional execution cost models.

    Returns impact in price-tick units.
    """
    if adv <= 0:
        return float("inf")
    participation = order_qty / adv
    return eta * volatility_daily * (participation ** beta)
