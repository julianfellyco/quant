"""
engine/slippage.py — SlippageModel: validated public interface for single-trade
                     cost estimation.

Why a separate class from compute_transaction_costs()?
-------------------------------------------------------
`compute_transaction_costs()` is the vectorised hot path for the backtesting
engine.  It operates on pl.Series, accepts signed trade sizes (positive = buy,
negative = sell), and is designed for throughput, not ergonomics.

`SlippageModel` is the public-facing API for:
  - Pre-trade cost estimation ("what will this trade cost?")
  - Unit testing with deterministic, injectable market data
  - Validation at system boundaries (negative sizes, unknown tickers)

It decomposes the total cost into spread and impact components explicitly,
making assertions like "impact cost grows non-linearly with size" unambiguous.

Cost decomposition (single trade, no volume-regime adjustment)
--------------------------------------------------------------
  notional     = trade_size × price

  spread_cost  = notional × base_bps / 10_000
                 ← fixed fraction of notional; scales linearly with trade_size
                 ← represents half-spread + exchange/clearing fees

  impact_bps   = trade_size / base_liquidity × impact_coefficient
  impact_cost  = notional × impact_bps / 10_000
               = price × trade_size² × impact_coefficient / (base_lc × 10_000)
                 ← grows as trade_size²: this is the non-linearity

Why impact scales as trade_size²
---------------------------------
  impact_bps ∝ trade_size       (more shares consumed → deeper in the book)
  notional    ∝ trade_size       (more shares × same price)
  impact_cost = notional × impact_bps ∝ trade_size²

Doubling trade size → 4× impact cost, but only 2× spread cost.
At large sizes the impact term dominates and cost-per-share rises linearly.

ATR note
--------
ATR is included in MarketData to match the engine interface and allow future
regime-scaling extensions, but for single-bar fixed-ATR calculations the
atr_scalar = ATR/ATR = 1.0, so it does not change the result.  The test
suite passes ATR=$2.00 and asserts on cost values that are independent of
the scalar.
"""

from __future__ import annotations

from dataclasses import dataclass

from .costs import COST_PARAMS


# --------------------------------------------------------------------------- #
# Data containers                                                               #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MarketData:
    """
    Snapshot of market conditions at execution time.

    Attributes
    ----------
    price : float
        Mid or close price in USD.  Used as the execution price.
    atr : float
        Average True Range in USD (14-bar default in the engine).
        Included for interface symmetry with the vectorised cost model;
        see module docstring for why it does not affect single-bar estimates.
    """
    price: float
    atr:   float

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")
        if self.atr < 0:
            raise ValueError(f"atr must be non-negative, got {self.atr}")


@dataclass(frozen=True)
class SlippageResult:
    """
    Full cost decomposition for a single trade.

    Fields
    ------
    trade_size            : shares traded (0 → all costs zero)
    notional              : trade_size × price
    spread_cost           : fixed-fraction cost (exchange fees + half-spread)
    impact_cost           : market-impact cost (grows as trade_size²)
    total_cost            : spread_cost + impact_cost
    cost_per_share        : total_cost / trade_size  (0 when trade_size == 0)
    cost_bps              : total_cost / notional × 10_000  (0 when notional == 0)
    spread_cost_per_share : spread_cost / trade_size
    impact_cost_per_share : impact_cost / trade_size
                            grows linearly with trade_size — the key non-linearity
    """
    trade_size:             int
    notional:               float
    spread_cost:            float
    impact_cost:            float
    total_cost:             float
    cost_per_share:         float
    cost_bps:               float
    spread_cost_per_share:  float
    impact_cost_per_share:  float

    _ZERO: "SlippageResult | None" = None   # sentinel (set below)

    @classmethod
    def zero(cls) -> "SlippageResult":
        """Canonical zero-cost result for trade_size == 0."""
        return cls(
            trade_size            = 0,
            notional              = 0.0,
            spread_cost           = 0.0,
            impact_cost           = 0.0,
            total_cost            = 0.0,
            cost_per_share        = 0.0,
            cost_bps              = 0.0,
            spread_cost_per_share = 0.0,
            impact_cost_per_share = 0.0,
        )

    def __repr__(self) -> str:
        return (
            f"SlippageResult("
            f"size={self.trade_size:,}, "
            f"total=${self.total_cost:,.4f}, "
            f"spread=${self.spread_cost:,.4f}, "
            f"impact=${self.impact_cost:,.4f}, "
            f"cps=${self.cost_per_share:.6f}, "
            f"bps={self.cost_bps:.4f})"
        )


# --------------------------------------------------------------------------- #
# SlippageModel                                                                  #
# --------------------------------------------------------------------------- #

class SlippageModel:
    """
    Validated, single-trade slippage estimator.

    Parameters
    ----------
    ticker : str
        Must be a key in engine.costs.COST_PARAMS.
        Raises ValueError immediately at construction for unknown tickers.

    Usage
    -----
    >>> model = SlippageModel("PFE")
    >>> md    = MarketData(price=100.0, atr=2.0)
    >>> result = model.calculate(trade_size=10_000, market_data=md)
    >>> print(result)
    SlippageResult(size=10,000, total=$501.79, ...)
    """

    #: Tickers for which calibrated parameters exist.
    VALID_TICKERS: frozenset[str] = frozenset(COST_PARAMS)

    def __init__(self, ticker: str) -> None:
        if ticker not in self.VALID_TICKERS:
            raise ValueError(
                f"Unknown ticker '{ticker}'. "
                f"Valid tickers: {sorted(self.VALID_TICKERS)}"
            )
        self._ticker = ticker
        self._p      = COST_PARAMS[ticker]

    # ------------------------------------------------------------------ #
    # Core method                                                           #
    # ------------------------------------------------------------------ #

    def calculate(
        self,
        trade_size:  int,
        market_data: MarketData,
    ) -> SlippageResult:
        """
        Estimate full execution cost for a single trade.

        Parameters
        ----------
        trade_size : int
            Number of shares to trade.  Must be >= 0.
            Raises ValueError for negative values.
        market_data : MarketData
            Current price and ATR snapshot.

        Returns
        -------
        SlippageResult
            Decomposed cost: spread, impact, total, per-share, bps.

        Raises
        ------
        ValueError
            If trade_size < 0.
        """
        if trade_size < 0:
            raise ValueError(
                f"trade_size must be >= 0 (use absolute share count; "
                f"side is tracked separately). Got: {trade_size}"
            )

        if trade_size == 0:
            return SlippageResult.zero()

        p       = self._p
        price   = market_data.price
        notional = trade_size * price

        # ── Spread cost: linear in notional ─────────────────────────── #
        # base_bps represents half-spread + exchange/clearing fees.
        # Doubles when trade_size doubles (same fraction of a larger notional).
        spread_cost = notional * p["base_bps"] / 10_000

        # ── Market impact: quadratic in trade_size ───────────────────── #
        # impact_bps grows with trade_size because deeper book levels are
        # consumed, each at a worse price.  Combined with notional also
        # growing, impact_cost scales as trade_size².
        #
        # impact_cost_per_share = price × trade_size × impact_coeff / (lc × 10_000)
        # → grows linearly with trade_size (even though total is quadratic)
        #
        # atr_scalar = 1.0 for fixed-ATR single-bar calculations.
        # (In the vectorised engine, atr/median_atr produces regime scaling.)
        impact_bps  = trade_size / p["base_liquidity"] * p["impact_coefficient"]
        impact_cost = notional * impact_bps / 10_000   # × atr_scalar (= 1.0)

        total_cost = spread_cost + impact_cost

        return SlippageResult(
            trade_size            = trade_size,
            notional              = notional,
            spread_cost           = spread_cost,
            impact_cost           = impact_cost,
            total_cost            = total_cost,
            cost_per_share        = total_cost    / trade_size,
            cost_bps              = total_cost    / notional * 10_000,
            spread_cost_per_share = spread_cost  / trade_size,
            impact_cost_per_share = impact_cost  / trade_size,
        )

    # ------------------------------------------------------------------ #
    # Convenience                                                           #
    # ------------------------------------------------------------------ #

    @property
    def ticker(self) -> str:
        return self._ticker

    @property
    def base_bps(self) -> float:
        return self._p["base_bps"]

    @property
    def impact_coefficient(self) -> float:
        return self._p["impact_coefficient"]

    @property
    def base_liquidity(self) -> int:
        return self._p["base_liquidity"]

    def __repr__(self) -> str:
        return (
            f"SlippageModel({self._ticker}, "
            f"base_bps={self.base_bps}, "
            f"impact_coeff={self.impact_coefficient}, "
            f"lc={self.base_liquidity:,})"
        )
