"""backtester/risk/portfolio_risk.py — Portfolio-level risk constraints."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskLimits:
    """Portfolio risk constraints and circuit breakers.

    These limits are checked before each new trade is allowed.  The engine
    calls `check_new_trade()` and skips the signal if the trade would breach
    any limit.
    """

    max_portfolio_heat: float = 0.06        # max total risk as % of capital
    max_single_position: float = 0.20       # max single position as % of capital
    max_sector_exposure: float = 0.40       # max single sector as % of capital
    max_correlation_exposure: float = 0.60  # max correlated positions as % of capital
    max_daily_loss: float = 0.03            # circuit breaker: stop trading for the day
    max_drawdown_halt: float = 0.15         # circuit breaker: halt strategy

    def check_new_trade(
        self,
        current_positions: dict[str, float],  # ticker → notional USD
        new_ticker: str,
        new_notional: float,
        capital: float,
    ) -> tuple[bool, str]:
        """Return (allowed, reason).

        Args:
            current_positions: current open notionals (signed, USD)
            new_ticker:        ticker of the proposed trade
            new_notional:      proposed position notional (signed, USD)
            capital:           current portfolio capital

        Returns:
            (True, "ok")             if the trade is allowed
            (False, reason_string)   if a limit would be breached
        """
        if capital <= 0:
            return False, "Capital is zero or negative."

        if abs(new_notional) / capital > self.max_single_position:
            return False, (
                f"Single position {abs(new_notional)/capital:.1%} exceeds "
                f"max {self.max_single_position:.1%}."
            )

        total_exposure = (
            sum(abs(v) for v in current_positions.values()) + abs(new_notional)
        )
        if total_exposure / capital > 1.0:  # no leverage by default
            return False, (
                f"Total exposure {total_exposure/capital:.1%} exceeds 100%."
            )

        return True, "ok"

    def check_drawdown(self, current_drawdown: float) -> tuple[bool, str]:
        """Return (continue_trading, reason).

        Args:
            current_drawdown: current drawdown as a fraction (negative, e.g. -0.12)
        """
        if abs(current_drawdown) >= self.max_drawdown_halt:
            return False, (
                f"Drawdown halt triggered: {current_drawdown:.1%} ≥ "
                f"-{self.max_drawdown_halt:.1%}"
            )
        return True, "ok"
