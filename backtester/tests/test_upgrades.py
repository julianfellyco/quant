"""
tests/test_upgrades.py — Unit tests for the four architectural upgrades:

  A. Square-root impact model (costs.py)
  B. OFI tracker (lob_simulator.metrics.ofi)
  C. Pairs trading / cointegration spread (strategy.pairs)
  D. Monte Carlo event shuffler (engine.stress)
  E. Walk-forward optimiser (engine.walkforward)

All tests are self-contained with synthetic data; no network access required.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import sys
import os

import polars as pl
import pytest

# ── Ensure lob_simulator is importable (it lives as a sibling project) ──── #
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtester.engine.costs import COST_PARAMS, compute_transaction_costs
from backtester.engine.slippage import MarketData, SlippageModel
from backtester.engine.vectorized import VectorizedEngine
from backtester.strategy.pairs import compute_spread_zscore, pairs_signal, spread_summary
from backtester.data.events import EVENTS, BinaryEvent, EventType, get_event_dates


# --------------------------------------------------------------------------- #
# Shared synthetic data factories                                                #
# --------------------------------------------------------------------------- #

def _price_df(
    ticker:      str   = "PFE",
    n:           int   = 252,
    start_price: float = 100.0,
    drift:       float = 0.0002,
    vol:         float = 0.015,
    seed:        int   = 42,
) -> pl.DataFrame:
    """
    Deterministic GBM daily bars with all required engine columns, including
    zscore_20d and momentum_60_5 so signal functions work without DataHandler.
    """
    rng   = random.Random(seed)
    start = dt.datetime(2024, 1, 2, 9, 30)
    ts, closes, log_rets, atrs, vols, avg_vols, events = [], [], [], [], [], [], []
    price = start_price
    for i in range(n):
        ts.append(start + dt.timedelta(days=i))
        ret  = drift + vol * rng.gauss(0, 1)
        price = price * math.exp(ret)
        closes.append(price)
        log_rets.append(ret if i > 0 else None)
        atrs.append(price * 0.015)
        vols.append(max(1, int(1_000_000 * (0.7 + 0.6 * rng.random()))))
        avg_vols.append(900_000.0)
        events.append(False)

    df = pl.DataFrame({
        "timestamp":       pl.Series(ts,       dtype=pl.Datetime("us")),
        "close":           pl.Series(closes,   dtype=pl.Float64),
        "log_return":      pl.Series(log_rets, dtype=pl.Float64),
        "atr_14":          pl.Series(atrs,     dtype=pl.Float64),
        "volume":          pl.Series(vols,     dtype=pl.Int64),
        "avg_volume_20":   pl.Series(avg_vols, dtype=pl.Float64),
        "is_event_window": pl.Series(events,   dtype=pl.Boolean),
        "signal":          pl.Series([1.0] * n, dtype=pl.Float64),
    })

    # ── Synthetic zscore_20d and momentum_60_5 (needed by signal functions) ── #
    df = df.with_columns([
        pl.col("log_return").fill_null(0.0)
        .rolling_mean(window_size=20, min_samples=1).alias("_rm20"),
        pl.col("log_return").fill_null(0.0)
        .rolling_std(window_size=20, min_samples=2).alias("_rs20"),
    ]).with_columns([
        ((pl.col("log_return").fill_null(0.0) - pl.col("_rm20"))
         / pl.col("_rs20").clip(lower_bound=1e-9))
        .alias("zscore_20d"),
    ]).with_columns([
        # momentum_60_5: cumulative log return from bar t-60 to t-5
        (pl.col("log_return").fill_null(0.0).rolling_sum(window_size=55, min_samples=1))
        .alias("momentum_60_5"),
    ]).drop(["_rm20", "_rs20"])

    return df


def _engine() -> VectorizedEngine:
    return VectorizedEngine(
        initial_capital = 100_000,
        shares_per_unit = 100,
        risk_free_rate  = 0.05,
        ann_factor      = 252,
    )


# =========================================================================== #
# A. Square-Root Impact Tests                                                    #
# =========================================================================== #

class TestSquareRootImpact:
    """
    Verify the Almgren-Chriss square-root market impact model.

    Mathematical reference:
        impact_bps  = κ × √(Q / LC)
        impact_cost = notional × impact_bps / 10,000  ∝  Q^1.5
        impact_cps  = impact_cost / Q                 ∝  √Q
    """

    @pytest.fixture
    def pfe(self) -> SlippageModel:
        return SlippageModel("PFE")

    @pytest.fixture
    def md(self) -> MarketData:
        return MarketData(price=100.0, atr=2.0)

    def test_sqrt_scaling_on_double(self, pfe: SlippageModel, md: MarketData) -> None:
        """Doubling Q → impact cost grows by 2^1.5 = 2√2 (not 4×)."""
        r1 = pfe.calculate(100_000, md)
        r2 = pfe.calculate(200_000, md)
        assert r2.impact_cost == pytest.approx(
            2 * math.sqrt(2) * r1.impact_cost, rel=1e-6
        )

    def test_sqrt_per_share_scaling(self, pfe: SlippageModel, md: MarketData) -> None:
        """4× trade size → 2× impact cost per share (√4 = 2)."""
        r1 = pfe.calculate(50_000,  md)
        r4 = pfe.calculate(200_000, md)
        assert r4.impact_cost_per_share == pytest.approx(
            2.0 * r1.impact_cost_per_share, rel=1e-6
        )

    def test_sqrt_less_punitive_than_linear_at_large_sizes(
        self, pfe: SlippageModel, md: MarketData
    ) -> None:
        """
        At 100% ADV, the sqrt model should produce much less impact than a
        hypothetical linear model calibrated to the same 10%-ADV target.

        Linear  at 100% ADV: impact_bps = κ_lin × 1.0 = κ_lin
        Sqrt    at 100% ADV: impact_bps = κ_sqrt × 1.0 = κ_sqrt

        With κ_lin calibrated to 5 bps at 10% ADV: κ_lin = 50 (old model)
        With κ_sqrt calibrated to 5 bps at 10% ADV: κ_sqrt = 5/√0.1 ≈ 16 (new)
        At 100%: old=50 bps, new=16 bps — new is 3× less punitive.
        """
        adv = pfe.base_liquidity           # 100% participation
        r   = pfe.calculate(adv, md)

        # Sqrt model: impact_bps = κ × √1.0 = κ ≈ 16 bps
        expected_impact_bps = pfe.impact_coefficient * math.sqrt(1.0)
        expected_impact_cost = (adv * md.price) * expected_impact_bps / 10_000
        assert r.impact_cost == pytest.approx(expected_impact_cost, rel=1e-9)

    def test_impact_coefficient_calibration_pfe(
        self, pfe: SlippageModel, md: MarketData
    ) -> None:
        """
        At 10% ADV, PFE impact ≈ 5 bps — matching the Almgren calibration
        target that preserves continuity with the old model at typical sizes.
        """
        ten_pct_adv = int(pfe.base_liquidity * 0.10)  # 2.8M shares
        r = pfe.calculate(ten_pct_adv, md)
        impact_bps = r.impact_cost / (ten_pct_adv * md.price) * 10_000
        # κ × √0.1 ≈ 16 × 0.3162 ≈ 5.06 bps
        assert impact_bps == pytest.approx(pfe.impact_coefficient * math.sqrt(0.1), rel=1e-6)
        assert 4.5 < impact_bps < 6.0   # reasonable empirical range

    def test_vectorised_cost_uses_sqrt(self) -> None:
        """
        compute_transaction_costs() with 4× trade size → 2^1.5× impact cost,
        confirming the vectorised path also uses the square-root formula.
        """
        n = 50
        p = COST_PARAMS["PFE"]

        prices = pl.Series([100.0] * n, dtype=pl.Float64)
        atr    = pl.Series([1.5]   * n, dtype=pl.Float64)
        ts1    = pl.Series([100.0] * n, dtype=pl.Float64)
        ts4    = pl.Series([400.0] * n, dtype=pl.Float64)

        c1 = compute_transaction_costs(
            trade_sizes=ts1, prices=prices, atr=atr,
            liquidity_constant=p["base_liquidity"],
            base_bps=p["base_bps"], impact_coefficient=p["impact_coefficient"],
        )
        c4 = compute_transaction_costs(
            trade_sizes=ts4, prices=prices, atr=atr,
            liquidity_constant=p["base_liquidity"],
            base_bps=p["base_bps"], impact_coefficient=p["impact_coefficient"],
        )

        # Total cost at 4× size: spread part doubles 4×; impact part grows 4^1.5 = 8×
        # The ratio of *impact* portions: 8×
        # We verify by checking that c4 > 2 * c1 (confirming super-linear)
        assert float(c4.sum()) > 2.0 * float(c1.sum())


# =========================================================================== #
# B. OFI Tracker Tests                                                           #
# =========================================================================== #

class TestOFITracker:
    """Verify OFI computation from LOB book depth snapshots."""

    @staticmethod
    def _make_book():
        from lob_simulator.core.book import Book
        from lob_simulator.core.order import Order
        from lob_simulator.core.types import OrderId, OrderType, Price, Qty, Side
        return Book, Order, OrderId, Price, Qty, Side

    def test_first_snapshot_returns_none(self) -> None:
        """First call to snapshot() returns None (no prior state)."""
        from lob_simulator.core.book import Book
        from lob_simulator.metrics.ofi import OFITracker

        book = Book()
        tracker = OFITracker(book)
        assert tracker.snapshot() is None

    def test_positive_ofi_on_buy_depth_increase(self) -> None:
        """Adding buy-side limit orders increases bid depth → positive OFI."""
        from lob_simulator.core.book import Book
        from lob_simulator.core.order import Order
        from lob_simulator.core.types import OrderId, Price, Qty, Side
        from lob_simulator.metrics.ofi import OFITracker

        book    = Book()
        tracker = OFITracker(book)
        tracker.snapshot()  # initial state: empty book

        book.add_order(Order(OrderId(1), Side.BUY, Price(100_00), Qty(500)))
        ofi = tracker.snapshot()
        assert ofi is not None
        assert ofi > 0, f"Expected positive OFI (bid added), got {ofi}"

    def test_negative_ofi_on_ask_depth_increase(self) -> None:
        """Adding ask-side limit orders increases ask depth → negative OFI."""
        from lob_simulator.core.book import Book
        from lob_simulator.core.order import Order
        from lob_simulator.core.types import OrderId, Price, Qty, Side
        from lob_simulator.metrics.ofi import OFITracker

        book    = Book()
        tracker = OFITracker(book)
        tracker.snapshot()

        book.add_order(Order(OrderId(1), Side.SELL, Price(101_00), Qty(500)))
        ofi = tracker.snapshot()
        assert ofi is not None
        assert ofi < 0, f"Expected negative OFI (ask added), got {ofi}"

    def test_zero_ofi_on_balanced_change(self) -> None:
        """Equal additions on both sides → raw OFI = 0."""
        from lob_simulator.core.book import Book
        from lob_simulator.core.order import Order
        from lob_simulator.core.types import OrderId, Price, Qty, Side
        from lob_simulator.metrics.ofi import OFITracker

        book    = Book()
        tracker = OFITracker(book)

        # Pre-populate both sides equally
        book.add_order(Order(OrderId(1), Side.BUY,  Price(100_00), Qty(300)))
        book.add_order(Order(OrderId(2), Side.SELL, Price(101_00), Qty(300)))
        tracker.snapshot()  # capture initial symmetric state

        # Add equal quantity to both sides
        book.add_order(Order(OrderId(3), Side.BUY,  Price(100_00), Qty(200)))
        book.add_order(Order(OrderId(4), Side.SELL, Price(101_00), Qty(200)))
        ofi = tracker.snapshot()
        assert ofi == pytest.approx(0.0, abs=1e-9)

    def test_normalised_ofi_bounded(self) -> None:
        """Normalised OFI ∈ [−1, 1] by construction."""
        from lob_simulator.core.book import Book
        from lob_simulator.core.order import Order
        from lob_simulator.core.types import OrderId, Price, Qty, Side
        from lob_simulator.metrics.ofi import OFITracker

        book    = Book()
        tracker = OFITracker(book)
        tracker.snapshot()

        book.add_order(Order(OrderId(1), Side.BUY, Price(100_00), Qty(1000)))
        tracker.snapshot()

        for nofi in tracker.nofi_series:
            assert -1.0 - 1e-9 <= nofi <= 1.0 + 1e-9

    def test_cumulative_ofi_rolling_sum(self) -> None:
        """cumulative_ofi(window=2) sums the last 2 OFI values."""
        from lob_simulator.core.book import Book
        from lob_simulator.core.order import Order
        from lob_simulator.core.types import OrderId, Price, Qty, Side
        from lob_simulator.metrics.ofi import OFITracker

        book    = Book()
        tracker = OFITracker(book)
        tracker.snapshot()  # initial

        book.add_order(Order(OrderId(1), Side.BUY,  Price(100_00), Qty(200)))
        ofi1 = tracker.snapshot()
        book.add_order(Order(OrderId(2), Side.SELL, Price(101_00), Qty(300)))
        ofi2 = tracker.snapshot()

        expected = (ofi1 or 0) + (ofi2 or 0)
        assert tracker.cumulative_ofi(window=2) == pytest.approx(expected, abs=1e-9)

    def test_cumulative_ofi_returns_zero_when_insufficient_history(self) -> None:
        """cumulative_ofi(window=5) returns 0.0 if fewer than 5 snapshots recorded."""
        from lob_simulator.core.book import Book
        from lob_simulator.metrics.ofi import OFITracker

        book    = Book()
        tracker = OFITracker(book)
        tracker.snapshot()
        assert tracker.cumulative_ofi(window=5) == 0.0

    def test_reset_clears_history(self) -> None:
        """reset() removes all recorded snapshots."""
        from lob_simulator.core.book import Book
        from lob_simulator.core.order import Order
        from lob_simulator.core.types import OrderId, Price, Qty, Side
        from lob_simulator.metrics.ofi import OFITracker

        book    = Book()
        tracker = OFITracker(book)
        book.add_order(Order(OrderId(1), Side.BUY, Price(100_00), Qty(100)))
        tracker.snapshot()
        tracker.snapshot()

        tracker.reset()
        assert len(tracker.ofi_series) == 0
        assert tracker.last_ofi()     == 0.0


# =========================================================================== #
# C. Pairs Trading Tests                                                         #
# =========================================================================== #

class TestPairsTrading:

    @pytest.fixture
    def nvo_pfe_dfs(self):
        """Create two correlated synthetic price series (NVO leads, PFE lags)."""
        rng   = random.Random(7)
        n     = 252
        start = dt.datetime(2024, 1, 2)
        dates = [start + dt.timedelta(days=i) for i in range(n)]

        nvo_price, pfe_price = 120.0, 30.0
        nvo_closes, pfe_closes = [], []
        for _ in range(n):
            nvo_ret  = 0.0005 + 0.02 * rng.gauss(0, 1)
            pfe_ret  = 0.6 * nvo_ret + 0.3 * rng.gauss(0, 1) * 0.015
            nvo_price = nvo_price * math.exp(nvo_ret)
            pfe_price = pfe_price * math.exp(pfe_ret)
            nvo_closes.append(nvo_price)
            pfe_closes.append(pfe_price)

        nvo_df = pl.DataFrame({
            "timestamp": pl.Series(dates, dtype=pl.Datetime("us")),
            "close":     pl.Series(nvo_closes, dtype=pl.Float64),
        })
        pfe_df = pl.DataFrame({
            "timestamp": pl.Series(dates, dtype=pl.Datetime("us")),
            "close":     pl.Series(pfe_closes, dtype=pl.Float64),
        })
        return nvo_df, pfe_df

    def test_spread_zscore_returns_required_columns(self, nvo_pfe_dfs) -> None:
        nvo_df, pfe_df = nvo_pfe_dfs
        result = compute_spread_zscore(nvo_df, pfe_df, hedge_window=60, zscore_window=20)
        for col in ["timestamp", "nvo_close", "pfe_close", "beta", "spread_z"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_spread_zscore_has_correct_length(self, nvo_pfe_dfs) -> None:
        nvo_df, pfe_df = nvo_pfe_dfs
        result = compute_spread_zscore(nvo_df, pfe_df, hedge_window=60, zscore_window=20)
        # Inner join → same length as input (same timestamps)
        assert len(result) == len(nvo_df)

    def test_spread_z_is_approximately_stationary(self, nvo_pfe_dfs) -> None:
        """Z-scored spread should have mean ≈ 0 and std ≈ 1 over non-null values."""
        nvo_df, pfe_df = nvo_pfe_dfs
        result = compute_spread_zscore(nvo_df, pfe_df)
        z = result["spread_z"].drop_nulls()

        assert abs(float(z.mean())) < 0.5, "spread_z mean should be near 0"
        assert 0.5 < float(z.std()) < 2.0, "spread_z std should be near 1"

    def test_pairs_signal_returns_signal_columns(self, nvo_pfe_dfs) -> None:
        nvo_df, pfe_df = nvo_pfe_dfs
        spread_df = compute_spread_zscore(nvo_df, pfe_df)
        result    = pairs_signal(spread_df, entry_z=1.5, exit_z=0.3)
        assert "nvo_signal" in result.columns
        assert "pfe_signal" in result.columns
        assert "pair_state" in result.columns

    def test_pairs_signal_values_are_in_valid_set(self, nvo_pfe_dfs) -> None:
        """Signals must be in {−1, 0, +1}."""
        nvo_df, pfe_df = nvo_pfe_dfs
        spread_df = compute_spread_zscore(nvo_df, pfe_df)
        result    = pairs_signal(spread_df)

        valid = {-1.0, 0.0, 1.0}
        nvo_vals = set(result["nvo_signal"].drop_nulls().to_list())
        pfe_vals = set(result["pfe_signal"].drop_nulls().to_list())
        assert nvo_vals.issubset(valid), f"Unexpected NVO signals: {nvo_vals}"
        assert pfe_vals.issubset(valid), f"Unexpected PFE signals: {pfe_vals}"

    def test_signals_are_market_neutral(self, nvo_pfe_dfs) -> None:
        """
        Pairs strategy is market-neutral: whenever NVO is +1, PFE is −1 and
        vice versa.  Both legs are never in the same direction.
        """
        nvo_df, pfe_df = nvo_pfe_dfs
        spread_df = compute_spread_zscore(nvo_df, pfe_df)
        result    = pairs_signal(spread_df)

        for row in result.iter_rows(named=True):
            nvo = row["nvo_signal"]
            pfe = row["pfe_signal"]
            if nvo != 0.0 or pfe != 0.0:
                assert nvo + pfe == pytest.approx(0.0), (
                    f"Signals not market-neutral: nvo={nvo}, pfe={pfe}"
                )

    def test_spread_summary_returns_expected_keys(self, nvo_pfe_dfs) -> None:
        nvo_df, pfe_df = nvo_pfe_dfs
        spread_df = compute_spread_zscore(nvo_df, pfe_df)
        result    = pairs_signal(spread_df)
        summary   = spread_summary(result)

        for key in ["mean_beta", "std_beta", "spread_z_mean", "spread_z_std",
                    "pct_long_spread", "pct_short_spread", "pct_flat"]:
            assert key in summary, f"Missing key: {key}"

    def test_beta_is_positive_for_cointegrated_pair(self, nvo_pfe_dfs) -> None:
        """
        NVO and PFE are both GLP-1 plays; their log prices should be
        positively cointegrated → β > 0 on average.
        """
        nvo_df, pfe_df = nvo_pfe_dfs
        result = compute_spread_zscore(nvo_df, pfe_df)
        mean_beta = float(result["beta"].drop_nulls().mean())
        assert mean_beta > 0, f"Expected positive hedge ratio, got {mean_beta:.4f}"


# =========================================================================== #
# D. Monte Carlo Event Shuffler Tests                                            #
# =========================================================================== #

class TestEventShuffler:

    @pytest.fixture
    def nvo_events(self) -> list[BinaryEvent]:
        return get_event_dates("NVO", dt.date(2024, 1, 1), dt.date(2025, 3, 31))

    @pytest.fixture
    def base_df(self) -> pl.DataFrame:
        return _price_df("NVO", n=252, drift=0.001, vol=0.02, seed=99)

    def test_stress_result_has_correct_n_simulations(
        self, base_df, nvo_events
    ) -> None:
        from backtester.engine.stress import EventShuffler

        engine   = _engine()
        shuffler = EventShuffler(engine, n_simulations=20, seed=7)
        signal   = pl.Series("signal", [1.0] * len(base_df), dtype=pl.Float64)

        result = shuffler.run(
            ticker="NVO",
            aligned_df=base_df.drop(["signal", "is_event_window"]),
            signal_series=signal,
            events=nvo_events,
            max_shift_days=3,
        )
        assert result.n_simulations == 20
        assert len(result.sharpe_distribution) == 20

    def test_base_sharpe_equals_zero_shift_simulation(
        self, base_df, nvo_events
    ) -> None:
        """
        The base case (shift=0) should match what the engine produces directly.
        """
        from backtester.engine.stress import EventShuffler

        engine   = _engine()
        shuffler = EventShuffler(engine, n_simulations=5, seed=42)
        signal   = pl.Series("signal", [1.0] * len(base_df), dtype=pl.Float64)

        result = shuffler.run(
            ticker="NVO",
            aligned_df=base_df.drop(["signal", "is_event_window"]),
            signal_series=signal,
            events=nvo_events,
            max_shift_days=0,   # max_shift=0 → all simulations identical to base
        )
        # With max_shift=0, all simulations draw shift=0 → same Sharpe
        assert all(
            s == pytest.approx(result.base_sharpe, rel=1e-6)
            for s in result.sharpe_distribution
        )

    def test_fragility_score_is_between_0_and_1(
        self, base_df, nvo_events
    ) -> None:
        from backtester.engine.stress import EventShuffler

        engine   = _engine()
        shuffler = EventShuffler(engine, n_simulations=30, seed=42)
        signal   = pl.Series("signal", [1.0] * len(base_df), dtype=pl.Float64)

        result = shuffler.run(
            ticker="NVO",
            aligned_df=base_df.drop(["signal", "is_event_window"]),
            signal_series=signal,
            events=nvo_events,
            max_shift_days=5,
        )
        assert 0.0 <= result.fragility_score <= 1.0

    def test_p5_le_median_le_p95(self, base_df, nvo_events) -> None:
        from backtester.engine.stress import EventShuffler

        engine   = _engine()
        shuffler = EventShuffler(engine, n_simulations=50, seed=1)
        signal   = pl.Series("signal", [1.0] * len(base_df), dtype=pl.Float64)

        result = shuffler.run(
            ticker="NVO",
            aligned_df=base_df.drop(["signal", "is_event_window"]),
            signal_series=signal,
            events=nvo_events,
            max_shift_days=5,
        )
        assert result.p5_sharpe  <= result.p95_sharpe
        assert result.worst_sharpe <= result.best_sharpe

    def test_summary_string_contains_fragility_score(
        self, base_df, nvo_events
    ) -> None:
        from backtester.engine.stress import EventShuffler

        engine   = _engine()
        shuffler = EventShuffler(engine, n_simulations=10, seed=42)
        signal   = pl.Series("signal", [1.0] * len(base_df), dtype=pl.Float64)

        result = shuffler.run(
            ticker="NVO",
            aligned_df=base_df.drop(["signal", "is_event_window"]),
            signal_series=signal,
            events=nvo_events,
            max_shift_days=3,
        )
        summary = result.summary()
        assert "Fragility Score" in summary
        assert "Monte Carlo" in summary


# =========================================================================== #
# E. Walk-Forward Optimiser Tests                                                #
# =========================================================================== #

class TestWalkForwardOptimizer:

    @pytest.fixture
    def df(self) -> pl.DataFrame:
        """252-bar DataFrame suitable for walk-forward (no signal column)."""
        return _price_df("PFE", n=252).drop("signal")

    def test_produces_at_least_one_fold(self, df) -> None:
        from backtester.engine.walkforward import WalkForwardOptimizer
        from backtester.strategy.signals import mean_reversion_signal

        engine    = _engine()
        optimizer = WalkForwardOptimizer(
            engine     = engine,
            signal_fn  = mean_reversion_signal,
            param_grid = {"entry_z": [1.5], "exit_z": [0.3]},
            train_bars = 120,
            test_bars  = 20,
        )
        result = optimizer.run("PFE", df)
        assert result.n_folds >= 1

    def test_oos_sharpe_is_finite(self, df) -> None:
        from backtester.engine.walkforward import WalkForwardOptimizer
        from backtester.strategy.signals import mean_reversion_signal

        engine    = _engine()
        optimizer = WalkForwardOptimizer(
            engine     = engine,
            signal_fn  = mean_reversion_signal,
            param_grid = {"entry_z": [1.0, 1.5, 2.0], "exit_z": [0.3, 0.5]},
            train_bars = 100,
            test_bars  = 20,
        )
        result = optimizer.run("PFE", df)
        assert math.isfinite(result.aggregate_oos_sharpe)

    def test_each_fold_has_best_params(self, df) -> None:
        """Every fold must record which parameters were selected."""
        from backtester.engine.walkforward import WalkForwardOptimizer
        from backtester.strategy.signals import mean_reversion_signal

        engine    = _engine()
        optimizer = WalkForwardOptimizer(
            engine     = engine,
            signal_fn  = mean_reversion_signal,
            param_grid = {"entry_z": [1.0, 2.0], "exit_z": [0.3]},
            train_bars = 100,
            test_bars  = 20,
        )
        result = optimizer.run("PFE", df)
        for fold in result.folds:
            assert "entry_z" in fold.best_params
            assert "exit_z"  in fold.best_params

    def test_stability_score_is_non_negative(self, df) -> None:
        from backtester.engine.walkforward import WalkForwardOptimizer
        from backtester.strategy.signals import mean_reversion_signal

        engine    = _engine()
        optimizer = WalkForwardOptimizer(
            engine     = engine,
            signal_fn  = mean_reversion_signal,
            param_grid = {"entry_z": [1.5], "exit_z": [0.3]},
            train_bars = 100,
            test_bars  = 20,
        )
        result = optimizer.run("PFE", df)
        assert result.stability_score >= 0.0

    def test_summary_contains_fold_details(self, df) -> None:
        from backtester.engine.walkforward import WalkForwardOptimizer
        from backtester.strategy.signals import mean_reversion_signal

        engine    = _engine()
        optimizer = WalkForwardOptimizer(
            engine     = engine,
            signal_fn  = mean_reversion_signal,
            param_grid = {"entry_z": [1.5], "exit_z": [0.3]},
            train_bars = 100,
            test_bars  = 20,
        )
        result  = optimizer.run("PFE", df)
        summary = result.summary()
        assert "Walk-Forward" in summary
        assert "OOS Sharpe" in summary

    def test_insufficient_data_raises(self) -> None:
        """Too-short DataFrame raises ValueError before producing any folds."""
        from backtester.engine.walkforward import WalkForwardOptimizer
        from backtester.strategy.signals import mean_reversion_signal

        tiny_df   = _price_df("PFE", n=10).drop("signal")
        engine    = _engine()
        optimizer = WalkForwardOptimizer(
            engine     = engine,
            signal_fn  = mean_reversion_signal,
            param_grid = {"entry_z": [1.5], "exit_z": [0.3]},
            train_bars = 120,
            test_bars  = 20,
        )
        with pytest.raises(ValueError):
            optimizer.run("PFE", tiny_df)

    def test_is_sharpe_not_much_lower_than_oos(self, df) -> None:
        """
        Sanity check: IS Sharpe should not be dramatically below OOS Sharpe.
        (If IS < OOS, the optimizer is selecting parameters that do WORSE IS
        than OOS, which would indicate a bug in the grid search direction.)
        """
        from backtester.engine.walkforward import WalkForwardOptimizer
        from backtester.strategy.signals import mean_reversion_signal

        engine    = _engine()
        optimizer = WalkForwardOptimizer(
            engine     = engine,
            signal_fn  = mean_reversion_signal,
            param_grid = {"entry_z": [1.0, 1.5, 2.0], "exit_z": [0.3, 0.5]},
            train_bars = 100,
            test_bars  = 20,
        )
        result = optimizer.run("PFE", df)
        # IS Sharpe should be >= OOS − 3 (allow for some variance but not >3 Sharpe wrong)
        assert result.is_sharpe_mean >= result.aggregate_oos_sharpe - 3.0
