"""backtester/risk/kelly.py — Kelly criterion utilities."""
from __future__ import annotations

import math


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """Compute the full Kelly fraction.

    Kelly % = W/|L| - (1-W)/|W|   (simplified: W - (1-W)/R for R = W/L)

    Args:
        win_rate: probability of a winning trade (0 < win_rate < 1)
        avg_win:  average gain on winning trades (positive)
        avg_loss: average loss on losing trades (positive magnitude)

    Returns:
        Kelly fraction [0, 1].  Clamped to zero for negative expectancy.
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    r = avg_win / avg_loss
    f = win_rate - (1 - win_rate) / r
    return max(0.0, f)


def fractional_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """Fractional Kelly bet size.

    Quarter-Kelly (fraction=0.25) is standard in practice — it dramatically
    reduces variance while capturing ~75% of the geometric growth rate.

    Returns:
        Fractional Kelly fraction [0, 1].
    """
    return kelly_fraction(win_rate, avg_win, avg_loss) * fraction


def kelly_from_sharpe(sharpe: float, ann_factor: int = 252) -> float:
    """Approximate Kelly fraction from annualised Sharpe ratio.

    Under log-normal returns: f* ≈ Sharpe / √ann_factor  (per-bar Kelly).
    This is a commonly used approximation for continuous trading.

    Args:
        sharpe:     annualised Sharpe ratio
        ann_factor: trading bars per year (252 for daily)

    Returns:
        Per-bar Kelly fraction.
    """
    if ann_factor <= 0:
        return 0.0
    return max(0.0, sharpe / math.sqrt(ann_factor))
