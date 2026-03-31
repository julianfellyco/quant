"""backtester/risk/position_sizer.py — Position sizing strategies."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PositionSizer(Protocol):
    """Protocol for position sizing strategies."""

    def compute_size(
        self,
        capital: float,
        price: float,
        signal_strength: float,  # [-1, 1]
        volatility: float,       # annualized vol
        adv: float,              # average daily volume (shares)
    ) -> int:
        """Return number of shares to trade (signed)."""
        ...


class FixedFractional:
    """Risk a fixed % of capital per trade.

    Sizes positions so that a 2-ATR adverse move risks exactly `risk_pct`
    of capital.  Caps at `max_position_pct` of capital and 1% of ADV.
    """

    def __init__(self, risk_pct: float = 0.02, max_position_pct: float = 0.20) -> None:
        self.risk_pct = risk_pct
        self.max_position_pct = max_position_pct

    def compute_size(
        self,
        capital: float,
        price: float,
        signal_strength: float,
        volatility: float,
        adv: float,
    ) -> int:
        """Compute signed share count using fixed-fractional risk sizing.

        Derives stop distance as 2× the daily volatility proxy, then sizes
        the position so that hitting the stop loses exactly `risk_pct` of
        capital. The result is capped by both the max-position and ADV limits.

        Args:
            capital: Current portfolio value in USD.
            price: Current bar close price per share.
            signal_strength: Directional signal in [-1, 1]; scales the output.
            volatility: Annualised volatility estimate for the instrument.
            adv: Average daily volume in shares (used to cap at 1% of ADV).

        Returns:
            Signed number of shares to trade (positive = long, negative = short).
            Returns 0 if price or adv is non-positive.
        """
        if price <= 0 or adv <= 0:
            return 0
        risk_dollars = capital * self.risk_pct
        daily_vol = volatility / (252 ** 0.5)
        stop_distance = price * daily_vol * 2  # 2× daily vol as stop proxy
        if stop_distance <= 0:
            return 0
        raw_size = risk_dollars / stop_distance
        max_size = (capital * self.max_position_pct) / price
        adv_limit = adv * 0.01  # never exceed 1% of ADV
        return int(min(raw_size, max_size, adv_limit) * signal_strength)


class VolatilityTarget:
    """Target a specific annualised portfolio volatility.

    Scales notional exposure so that position vol equals `target_vol`.
    Caps at `max_leverage` × capital and 1% of ADV.
    """

    def __init__(self, target_vol: float = 0.15, max_leverage: float = 1.0) -> None:
        self.target_vol = target_vol
        self.max_leverage = max_leverage

    def compute_size(
        self,
        capital: float,
        price: float,
        signal_strength: float,
        volatility: float,
        adv: float,
    ) -> int:
        """Compute signed share count targeting a fixed annualised portfolio volatility.

        Notional = capital × (target_vol / instrument_vol), then capped by
        max_leverage and ADV limits. Instruments with higher realised volatility
        receive smaller notional allocations so the portfolio contribution stays constant.

        Args:
            capital: Current portfolio value in USD.
            price: Current bar close price per share.
            signal_strength: Directional signal in [-1, 1]; scales the output.
            volatility: Annualised volatility estimate for the instrument.
            adv: Average daily volume in shares (used to cap at 1% of ADV).

        Returns:
            Signed number of shares to trade (positive = long, negative = short).
            Returns 0 if volatility, price, or adv is non-positive.
        """
        if volatility <= 0 or price <= 0 or adv <= 0:
            return 0
        target_notional = capital * (self.target_vol / volatility)
        max_notional = capital * self.max_leverage
        notional = min(target_notional, max_notional)
        raw_size = notional / price
        adv_limit = adv * 0.01
        return int(min(raw_size, adv_limit) * signal_strength)


class KellyCriterion:
    """Full and fractional Kelly sizing.

    Uses empirical win-rate and average win/loss ratio to compute the
    Kelly fraction, then applies `fraction` (default 0.25 = quarter-Kelly)
    to reduce bet size and limit variance.
    """

    def __init__(
        self,
        fraction: float = 0.25,
        max_position_pct: float = 0.25,
        win_rate: float = 0.55,
        avg_win_loss_ratio: float = 1.5,
    ) -> None:
        self.fraction = fraction
        self.max_position_pct = max_position_pct
        self.win_rate = win_rate
        self.avg_win_loss_ratio = avg_win_loss_ratio

    def compute_size(
        self,
        capital: float,
        price: float,
        signal_strength: float,
        volatility: float,
        adv: float,
    ) -> int:
        """Compute signed share count using fractional Kelly criterion.

        Kelly % = W - (1 - W) / R, where W is win_rate and R is avg_win_loss_ratio.
        The raw Kelly fraction is multiplied by `self.fraction` (e.g. 0.25 for
        quarter-Kelly) to dampen variance. Notional is further capped by
        max_position_pct and 1% of ADV.

        Args:
            capital: Current portfolio value in USD.
            price: Current bar close price per share.
            signal_strength: Directional signal in [-1, 1]; scales the output.
            volatility: Annualised volatility estimate (not used directly, accepted
                for Protocol compatibility).
            adv: Average daily volume in shares (used to cap at 1% of ADV).

        Returns:
            Signed number of shares to trade (positive = long, negative = short).
            Returns 0 if price or adv is non-positive, or if the Kelly fraction is zero.
        """
        if price <= 0 or adv <= 0:
            return 0
        # Kelly % = W - (1-W)/R
        kelly_pct = self.win_rate - ((1 - self.win_rate) / self.avg_win_loss_ratio)
        kelly_pct = max(0.0, kelly_pct) * self.fraction
        if kelly_pct <= 0:
            return 0
        notional = capital * kelly_pct
        max_notional = capital * self.max_position_pct
        raw_size = min(notional, max_notional) / price
        adv_limit = adv * 0.01
        return int(min(raw_size, adv_limit) * signal_strength)
