"""backtester/engine/lob_executor.py — LOB-based execution simulator.

Simulates order execution through a synthetic limit order book rather than
applying static slippage.  Gives more realistic fill prices for large orders
by walking the book level-by-level.

Execution model
---------------
1. Build a synthetic book centred on mid_price with `levels` price levels.
2. Depth at each level decays geometrically: base_depth × depth_decay^i.
3. Walk the appropriate side (ask for buys, bid for sells) to fill the order.
4. Return volume-weighted average fill price, slippage bps, and fill rate.

Integration
-----------
LOBExecutor can be used instead of the Almgren-Chriss cost model:
    executor = LOBExecutor()
    fill = executor.simulate_fill(mid_price=100.0, quantity=500, spread_bps=3.0)
    cost_usd = fill["slippage_bps"] / 10_000 * abs(quantity) * mid_price
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FillResult:
    """Result of a simulated LOB execution."""

    fill_price:         float  # volume-weighted average fill price
    slippage_bps:       float  # bps vs fair (mid + half-spread) price
    market_impact_bps:  float  # bps vs mid price
    fill_rate:          float  # fraction of order filled (0–1)


class LOBExecutor:
    """Simulates order execution through a synthetic limit order book.

    Args:
        levels:      Number of price levels on each side of the book.
        base_depth:  Shares at the best level.
        tick_size:   Minimum price increment.
        depth_decay: Geometric decay of depth away from best (0 < d < 1).
    """

    def __init__(
        self,
        levels: int = 10,
        base_depth: int = 1_000,
        tick_size: float = 0.01,
        depth_decay: float = 0.8,
    ) -> None:
        self.levels = levels
        self.base_depth = base_depth
        self.tick_size = tick_size
        self.depth_decay = depth_decay

    def simulate_fill(
        self,
        mid_price: float,
        quantity: int,       # positive = buy, negative = sell
        spread_bps: float = 2.0,
    ) -> FillResult:
        """Simulate filling a market order through a synthetic book.

        Args:
            mid_price:  Current mid price.
            quantity:   Signed shares to fill (positive = buy, negative = sell).
            spread_bps: Bid-ask spread in basis points.

        Returns:
            FillResult with fill price, slippage, impact, and fill rate.
        """
        if quantity == 0 or mid_price <= 0:
            return FillResult(
                fill_price=mid_price, slippage_bps=0.0,
                market_impact_bps=0.0, fill_rate=0.0,
            )

        half_spread = mid_price * (spread_bps / 10_000) / 2
        is_buy = quantity > 0
        remaining = abs(quantity)

        # Build the side we need to walk
        # Buy order walks the ask side (prices ascending from mid + half_spread)
        # Sell order walks the bid side (prices descending from mid - half_spread)
        levels_data: list[tuple[float, int]] = []
        for i in range(self.levels):
            depth = int(self.base_depth * (self.depth_decay ** i))
            if depth < 1:
                break
            if is_buy:
                price = round(mid_price + half_spread + i * self.tick_size, 4)
            else:
                price = round(mid_price - half_spread - i * self.tick_size, 4)
            levels_data.append((price, depth))

        # Walk the book
        total_cost = 0.0
        filled = 0
        for price, depth in levels_data:
            can_fill = min(remaining, depth)
            total_cost += can_fill * price
            filled += can_fill
            remaining -= can_fill
            if remaining <= 0:
                break

        if filled == 0:
            return FillResult(
                fill_price=mid_price, slippage_bps=0.0,
                market_impact_bps=0.0, fill_rate=0.0,
            )

        avg_fill = total_cost / filled
        fair_price = mid_price + (half_spread if is_buy else -half_spread)

        slippage_bps = abs(avg_fill - fair_price) / mid_price * 10_000
        impact_bps   = abs(avg_fill - mid_price)  / mid_price * 10_000

        return FillResult(
            fill_price=        round(avg_fill, 6),
            slippage_bps=      round(slippage_bps, 3),
            market_impact_bps= round(impact_bps, 3),
            fill_rate=         round(filled / abs(quantity), 4),
        )

    def cost_usd(
        self,
        mid_price: float,
        quantity: int,
        spread_bps: float = 2.0,
    ) -> float:
        """Convenience: return total execution cost in USD.

        Suitable for direct substitution into the transaction cost pipeline.

        Args:
            mid_price:  Bar close price used as mid.
            quantity:   Signed shares traded.
            spread_bps: Effective bid-ask spread.

        Returns:
            Non-negative cost in USD.
        """
        if quantity == 0:
            return 0.0
        fill = self.simulate_fill(mid_price, quantity, spread_bps)
        return abs(fill.slippage_bps / 10_000 * abs(quantity) * mid_price)
