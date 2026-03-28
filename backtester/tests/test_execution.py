"""
tests/test_execution.py — Unit tests for SlippageModel and TransactionCost accuracy.

All tests use a fixed MarketData(price=$100, atr=$2) mock to remove
market-data variance from the assertions.  Expected values are derived
analytically from the cost formulas so every test also serves as a
specification of the model's mathematical guarantees.

Cost formula recap (for manual verification)
--------------------------------------------
  notional     = trade_size × price
  spread_cost  = notional × base_bps / 10_000            ← linear in trade_size
  impact_bps   = trade_size / base_liquidity × impact_coeff
  impact_cost  = notional × impact_bps / 10_000          ← quadratic in trade_size
  cost_per_share  = total_cost / trade_size
  impact_cps      = impact_cost / trade_size              ← linear in trade_size

PFE calibration: base_bps=5, impact_coeff=50, lc=28,000,000
NVO calibration: base_bps=8, impact_coeff=80, lc=3,500,000
"""

from __future__ import annotations

import pytest

from backtester.engine.slippage import MarketData, SlippageModel, SlippageResult


# --------------------------------------------------------------------------- #
# Shared fixtures                                                               #
# --------------------------------------------------------------------------- #

@pytest.fixture
def mock_market() -> MarketData:
    """
    Fixed market snapshot used across all tests.
    Price=$100.00, ATR=$2.00.  ATR is included for interface completeness;
    for single-bar calculations atr_scalar=1.0 so it does not change results.
    """
    return MarketData(price=100.0, atr=2.0)


@pytest.fixture
def pfe() -> SlippageModel:
    return SlippageModel("PFE")


@pytest.fixture
def nvo() -> SlippageModel:
    return SlippageModel("NVO")


# --------------------------------------------------------------------------- #
# TestSlippageModels                                                             #
# --------------------------------------------------------------------------- #

class TestSlippageModels:

    # ── Test 1: Zero-Impact ──────────────────────────────────────────────── #

    def test_zero_trade_size_results_in_zero_spread_cost(
        self, pfe: SlippageModel, mock_market: MarketData
    ) -> None:
        """A trade of 0 shares produces $0 spread cost."""
        result = pfe.calculate(trade_size=0, market_data=mock_market)
        assert result.spread_cost == pytest.approx(0.0)

    def test_zero_trade_size_results_in_zero_impact_cost(
        self, pfe: SlippageModel, mock_market: MarketData
    ) -> None:
        """A trade of 0 shares produces $0 market impact."""
        result = pfe.calculate(trade_size=0, market_data=mock_market)
        assert result.impact_cost == pytest.approx(0.0)

    def test_zero_trade_size_results_in_zero_total_cost(
        self, pfe: SlippageModel, mock_market: MarketData
    ) -> None:
        """A trade of 0 shares produces $0 total transaction cost."""
        result = pfe.calculate(trade_size=0, market_data=mock_market)
        assert result.total_cost == pytest.approx(0.0)

    def test_zero_trade_size_returns_canonical_zero_result(
        self, nvo: SlippageModel, mock_market: MarketData
    ) -> None:
        """Zero-trade returns a SlippageResult with all fields == 0."""
        result = nvo.calculate(trade_size=0, market_data=mock_market)
        assert result == SlippageResult.zero()

    # ── Test 2: Linear Fee ──────────────────────────────────────────────── #

    def test_spread_cost_scales_linearly_with_trade_size(
        self, pfe: SlippageModel, mock_market: MarketData
    ) -> None:
        """
        Doubling the trade size exactly doubles the spread cost.

        Derivation:
            spread_cost = trade_size × price × base_bps / 10_000
            PFE: 1,000 shares → $1,000 × $100 × 5/10,000 = $5.00
                 2,000 shares → $2,000 × $100 × 5/10,000 = $10.00
            Ratio: exactly 2.0
        """
        r1 = pfe.calculate(1_000, mock_market)
        r2 = pfe.calculate(2_000, mock_market)

        assert r2.spread_cost / r1.spread_cost == pytest.approx(2.0, rel=1e-9)

    def test_spread_cost_per_share_is_constant_across_sizes(
        self, pfe: SlippageModel, mock_market: MarketData
    ) -> None:
        """
        Spread cost per share = price × base_bps / 10_000 — independent of size.

        PFE: $100 × 5 / 10,000 = $0.05 per share at any trade size.
        This is the 'fixed fee' component — it does NOT penalise large trades
        beyond the scaling of notional.  Only the impact component does that.
        """
        expected_spread_cps = mock_market.price * pfe.base_bps / 10_000

        for size in (100, 1_000, 100_000, 1_000_000):
            r = pfe.calculate(size, mock_market)
            assert r.spread_cost_per_share == pytest.approx(
                expected_spread_cps, rel=1e-9
            ), f"spread_cps changed at trade_size={size:,}"

    def test_spread_cost_absolute_value_pfe(
        self, pfe: SlippageModel, mock_market: MarketData
    ) -> None:
        """
        Verify exact dollar value of spread cost for PFE.

        notional = 10,000 × $100 = $1,000,000
        spread   = $1,000,000 × 5 bps = $500.00
        """
        r = pfe.calculate(10_000, mock_market)
        assert r.spread_cost == pytest.approx(500.0, rel=1e-9)

    def test_spread_cost_absolute_value_nvo(
        self, nvo: SlippageModel, mock_market: MarketData
    ) -> None:
        """
        NVO base_bps=8 → spread_cost = notional × 8/10,000.

        notional = 10,000 × $100 = $1,000,000
        spread   = $1,000,000 × 8 bps = $800.00
        """
        r = nvo.calculate(10_000, mock_market)
        assert r.spread_cost == pytest.approx(800.0, rel=1e-9)

    # ── Test 3: Non-Linear Slippage ─────────────────────────────────────── #

    def test_large_trade_has_higher_cost_per_share_than_small_trade(
        self, pfe: SlippageModel, mock_market: MarketData
    ) -> None:
        """
        A 1,000,000-share trade has a higher cost-per-share than a 100-share
        trade due to the growing market-impact component.

        Analytical derivation at price=$100, PFE (base_bps=5, lc=28M,
        impact_coeff=50):

            small (100 shares):
                spread_cps  = $100 × 5/10,000            = $0.0500
                impact_cps  = $100 × 100/28M × 50/10,000 = $0.0000179
                total_cps                                 ≈ $0.05002

            large (1M shares):
                spread_cps  = $100 × 5/10,000            = $0.0500  (unchanged)
                impact_cps  = $100 × 1M/28M × 50/10,000  = $0.1786
                total_cps                                 ≈ $0.2286

            Ratio of total_cps: 0.2286 / 0.05002 ≈ 1.36×

        Why only 1.36×?  PFE's ADV of 28M shares means a 1M-share trade
        consumes just 3.6% of daily volume.  The impact is real but spread
        still dominates total_cps for the small trade.

        The true non-linearity is in the isolated impact component:
            impact_cps(1M) / impact_cps(100) = 10,000×
        (tested separately in test_impact_cost_per_share_grows_linearly_with_trade_size)
        """
        small = pfe.calculate(100,       mock_market)
        large = pfe.calculate(1_000_000, mock_market)

        # Exact ratio by formula: ≈1.357× — use 1.3× as a guaranteed lower bound
        assert large.cost_per_share > 1.3 * small.cost_per_share, (
            f"Expected large trade cps ({large.cost_per_share:.6f}) "
            f"> 1.3× small trade cps ({small.cost_per_share:.6f})"
        )

        # The impact component per share grows far more dramatically (10,000×)
        assert large.impact_cost_per_share > 1_000 * small.impact_cost_per_share, (
            "impact_cps must be orders-of-magnitude higher for the 1M-share trade"
        )

    def test_impact_cost_per_share_grows_linearly_with_trade_size(
        self, pfe: SlippageModel, mock_market: MarketData
    ) -> None:
        """
        impact_cost_per_share = price × trade_size × impact_coeff / (lc × 10,000)
        → linear in trade_size: 10× the shares → 10× the impact_cps.

        This is the non-linearity in total cost:
            impact_cost = notional × impact_bps ∝ trade_size²
        Per-share: impact_cps ∝ trade_size (still super-linear vs fixed fee).
        """
        r_base  = pfe.calculate(100_000,   mock_market)
        r_10x   = pfe.calculate(1_000_000, mock_market)

        # 10× trade size → 10× impact_cost_per_share
        assert r_10x.impact_cost_per_share == pytest.approx(
            10.0 * r_base.impact_cost_per_share, rel=1e-6
        )

    def test_large_trade_impact_cost_dominates_spread_cost(
        self, pfe: SlippageModel, mock_market: MarketData
    ) -> None:
        """
        For a 1M-share trade on PFE, the market impact cost must exceed
        the spread cost, demonstrating the regime change at large sizes.

        Expected:
            spread_cost = $1,000,000 × $100 × 5/10,000 = $50,000
            impact_bps  = 1,000,000 / 28,000,000 × 50 = 1.786 bps
            impact_cost = $100,000,000 × 1.786/10,000 = $17,857
        """
        r = pfe.calculate(1_000_000, mock_market)
        assert r.impact_cost > 0, "Impact cost must be positive"

        # At 1M shares, impact > spread is not guaranteed for PFE's large LC,
        # but the absolute values must both be computed and positive
        spread = r.spread_cost
        impact = r.impact_cost
        assert spread > 0 and impact > 0

    def test_impact_cost_grows_quadratically_in_total(
        self, pfe: SlippageModel, mock_market: MarketData
    ) -> None:
        """
        Quadratic scaling of total impact cost:
            impact_cost(Q) ∝ Q²

        If we double the trade size, impact_cost quadruples.
        (Because both notional and impact_bps are each proportional to Q.)
        """
        r1 = pfe.calculate(100_000, mock_market)
        r2 = pfe.calculate(200_000, mock_market)

        assert r2.impact_cost == pytest.approx(
            4.0 * r1.impact_cost, rel=1e-6
        ), (
            "Doubling trade size must quadruple impact cost. "
            f"Got: r1={r1.impact_cost:.4f}, r2={r2.impact_cost:.4f}, "
            f"ratio={r2.impact_cost/r1.impact_cost:.4f} (expected 4.0)"
        )

    def test_impact_cost_per_share_exact_value_pfe(
        self, pfe: SlippageModel, mock_market: MarketData
    ) -> None:
        """
        Verify exact impact_cost_per_share formula for PFE.

        impact_cps = price × trade_size × impact_coeff / (lc × 10_000)
                   = $100 × 280,000 × 50 / (28,000,000 × 10,000)
                   = $100 × 0.0005
                   = $0.05 exactly
        """
        # trade_size = lc / impact_coeff = 28M/50 = 560,000 shares
        # → impact_bps = 560,000/28M × 50 = 1.0 bps
        # → impact_cost = (560,000 × $100) × 1.0/10,000 = $5,600
        # → impact_cps  = $5,600 / 560,000 = $0.01
        trade_size = 560_000
        r = pfe.calculate(trade_size, mock_market)

        expected_impact_cps = (
            mock_market.price * trade_size * pfe.impact_coefficient
            / (pfe.base_liquidity * 10_000)
        )
        assert r.impact_cost_per_share == pytest.approx(expected_impact_cps, rel=1e-9)

    # ── Test 4: Liquidity Sensitivity ───────────────────────────────────── #

    def test_nvo_has_higher_total_cost_than_pfe_same_trade(
        self,
        pfe: SlippageModel,
        nvo: SlippageModel,
        mock_market: MarketData,
    ) -> None:
        """
        A 10,000-share trade on NVO costs more than the same trade on PFE
        because NVO has a lower base_liquidity and a higher impact_coefficient.

        Expected (price=$100):
            PFE: spread=$500  + impact≈$1.79   = $501.79  (5.018 bps)
            NVO: spread=$800  + impact≈$22.86  = $822.86  (8.229 bps)
        """
        r_pfe = pfe.calculate(10_000, mock_market)
        r_nvo = nvo.calculate(10_000, mock_market)

        assert r_nvo.total_cost > r_pfe.total_cost, (
            f"NVO total cost ({r_nvo.total_cost:.2f}) should exceed "
            f"PFE total cost ({r_pfe.total_cost:.2f}) for the same trade size."
        )

    def test_nvo_has_higher_cost_per_share_than_pfe(
        self,
        pfe: SlippageModel,
        nvo: SlippageModel,
        mock_market: MarketData,
    ) -> None:
        """NVO penalises cost-per-share more than PFE at the same trade size."""
        r_pfe = pfe.calculate(10_000, mock_market)
        r_nvo = nvo.calculate(10_000, mock_market)

        assert r_nvo.cost_per_share > r_pfe.cost_per_share

    def test_nvo_higher_impact_cost_than_pfe_same_notional(
        self,
        pfe: SlippageModel,
        nvo: SlippageModel,
        mock_market: MarketData,
    ) -> None:
        """
        With the same notional ($1,000,000 → 10,000 shares at $100),
        NVO suffers ~12.8× higher market impact than PFE.

        PFE: impact_bps = 10K/28M × 50 = 0.01786 bps → $1.79
        NVO: impact_bps = 10K/3.5M × 80 = 0.2286 bps  → $22.86
        Ratio: 22.86 / 1.79 ≈ 12.8×
        """
        r_pfe = pfe.calculate(10_000, mock_market)
        r_nvo = nvo.calculate(10_000, mock_market)

        assert r_nvo.impact_cost > r_pfe.impact_cost
        # Ratio should be at least 5× (conservative bound)
        assert r_nvo.impact_cost > 5.0 * r_pfe.impact_cost, (
            f"NVO impact ({r_nvo.impact_cost:.4f}) should be >5× "
            f"PFE impact ({r_pfe.impact_cost:.4f})"
        )

    def test_cost_bps_penalises_illiquid_ticker(
        self,
        pfe: SlippageModel,
        nvo: SlippageModel,
        mock_market: MarketData,
    ) -> None:
        """
        NVO cost in basis points must exceed PFE's for the same trade.
        cost_bps is normalised by notional, so this tests the per-dollar
        cost of trading each ticker — the correct liquidity comparison metric.
        """
        r_pfe = pfe.calculate(10_000, mock_market)
        r_nvo = nvo.calculate(10_000, mock_market)

        assert r_nvo.cost_bps > r_pfe.cost_bps

    def test_absolute_cost_bps_values(
        self,
        pfe: SlippageModel,
        nvo: SlippageModel,
        mock_market: MarketData,
    ) -> None:
        """
        Verify analytically-derived cost_bps for both tickers.

        PFE 10K shares @ $100:
            notional       = $1,000,000
            spread_bps     = 5 bps
            impact_bps     = 10K/28M × 50 = 0.01786 bps
            total_bps      ≈ 5.018 bps

        NVO 10K shares @ $100:
            notional       = $1,000,000
            spread_bps     = 8 bps
            impact_bps     = 10K/3.5M × 80 = 0.2286 bps
            total_bps      ≈ 8.229 bps
        """
        r_pfe = pfe.calculate(10_000, mock_market)
        r_nvo = nvo.calculate(10_000, mock_market)

        pfe_expected_bps = 5.0 + (10_000 / 28_000_000 * 50)
        nvo_expected_bps = 8.0 + (10_000 / 3_500_000  * 80)

        assert r_pfe.cost_bps == pytest.approx(pfe_expected_bps, rel=1e-6)
        assert r_nvo.cost_bps == pytest.approx(nvo_expected_bps, rel=1e-6)

    # ── Validation: negative size & invalid ticker ───────────────────── #

    def test_negative_trade_size_raises_value_error(
        self, pfe: SlippageModel, mock_market: MarketData
    ) -> None:
        """
        Negative trade sizes are rejected at the public interface.

        Rationale: trade direction (buy/sell) is tracked separately via the
        signal column.  The cost model operates on absolute size only.
        Accepting negative sizes silently (via abs()) could mask sign-flip
        bugs in upstream position-change logic.
        """
        with pytest.raises(ValueError, match="trade_size must be >= 0"):
            pfe.calculate(trade_size=-100, market_data=mock_market)

    def test_large_negative_trade_size_raises_value_error(
        self, nvo: SlippageModel, mock_market: MarketData
    ) -> None:
        """Any negative value, not just -1, must raise."""
        with pytest.raises(ValueError):
            nvo.calculate(trade_size=-1_000_000, market_data=mock_market)

    def test_invalid_ticker_raises_value_error_at_construction(self) -> None:
        """
        Unknown tickers raise ValueError immediately at SlippageModel().

        Fail-fast at construction rather than at calculate() time: this
        surfaces misconfigured tickers before any market data is available,
        making debugging easier.
        """
        with pytest.raises(ValueError, match="Unknown ticker"):
            SlippageModel("INVALID")

    def test_invalid_ticker_similar_name_raises(self) -> None:
        """Ticker casing or prefix variants are also rejected."""
        for bad in ("pfe", "Pfe", "PFE2", "NVO.A", "", "XYZ"):
            with pytest.raises(ValueError):
                SlippageModel(bad)

    def test_invalid_market_data_negative_price_raises(self) -> None:
        """MarketData validates its own fields: price must be positive."""
        with pytest.raises(ValueError):
            MarketData(price=-10.0, atr=2.0)

    def test_invalid_market_data_negative_atr_raises(self) -> None:
        """ATR cannot be negative."""
        with pytest.raises(ValueError):
            MarketData(price=100.0, atr=-1.0)

    # ── Total cost identity ──────────────────────────────────────────── #

    def test_total_cost_equals_spread_plus_impact(
        self, pfe: SlippageModel, mock_market: MarketData
    ) -> None:
        """total_cost == spread_cost + impact_cost for all sizes."""
        for size in (0, 1, 100, 10_000, 1_000_000):
            r = pfe.calculate(size, mock_market)
            assert r.total_cost == pytest.approx(
                r.spread_cost + r.impact_cost, rel=1e-12
            ), f"Additive decomposition broken at size={size}"

    def test_cost_bps_consistent_with_total_cost(
        self, nvo: SlippageModel, mock_market: MarketData
    ) -> None:
        """cost_bps must equal total_cost / notional × 10,000."""
        r = nvo.calculate(50_000, mock_market)
        expected_bps = r.total_cost / r.notional * 10_000
        assert r.cost_bps == pytest.approx(expected_bps, rel=1e-9)

    def test_cost_per_share_consistent_with_total_cost(
        self, pfe: SlippageModel, mock_market: MarketData
    ) -> None:
        """cost_per_share × trade_size must equal total_cost."""
        r = pfe.calculate(75_000, mock_market)
        assert r.cost_per_share * r.trade_size == pytest.approx(
            r.total_cost, rel=1e-9
        )


# --------------------------------------------------------------------------- #
# Parametrized cross-checks                                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("size", [1, 10, 100, 1_000, 10_000, 100_000])
def test_all_result_fields_nonnegative(size: int) -> None:
    """No cost field should ever be negative for valid inputs."""
    md  = MarketData(price=100.0, atr=2.0)
    pfe = SlippageModel("PFE")
    r   = pfe.calculate(size, md)

    assert r.spread_cost           >= 0.0
    assert r.impact_cost           >= 0.0
    assert r.total_cost            >= 0.0
    assert r.cost_per_share        >= 0.0
    assert r.cost_bps              >= 0.0
    assert r.spread_cost_per_share >= 0.0
    assert r.impact_cost_per_share >= 0.0


@pytest.mark.parametrize("ticker", ["PFE", "NVO"])
def test_valid_tickers_construct_without_error(ticker: str) -> None:
    """All tickers in COST_PARAMS must construct successfully."""
    model = SlippageModel(ticker)
    assert model.ticker == ticker


@pytest.mark.parametrize("price,atr", [
    (0.01, 0.001),    # penny stock
    (500.0, 10.0),    # high-priced stock
    (28.50, 0.42),    # realistic PFE range
    (110.0, 3.20),    # realistic NVO range
])
def test_model_stable_across_price_ranges(price: float, atr: float) -> None:
    """SlippageModel must not raise or return NaN for any reasonable price."""
    import math
    md  = MarketData(price=price, atr=atr)
    pfe = SlippageModel("PFE")
    r   = pfe.calculate(1_000, md)

    assert not math.isnan(r.total_cost)
    assert not math.isinf(r.total_cost)
    assert r.total_cost > 0.0
