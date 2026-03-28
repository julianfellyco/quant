"""
tests/test_engine.py — Unit tests for costs, VectorizedEngine, and statistics.

All tests use synthetic deterministic data so they run offline and produce
reproducible results regardless of market data availability.
"""

from __future__ import annotations

import datetime as dt
import math
import random

import polars as pl
import pytest

from backtester.engine.costs import compute_transaction_costs, cost_for_ticker
from backtester.engine.vectorized import VectorizedEngine, BacktestResult
from backtester.engine.report import build_comparison_table
from backtester.stats.metrics import compute_event_window_report


# --------------------------------------------------------------------------- #
# Synthetic data factories                                                       #
# --------------------------------------------------------------------------- #

def _price_df(
    n:           int   = 252,
    start_price: float = 100.0,
    drift:       float = 0.0005,
    vol:         float = 0.015,
    seed:        int   = 42,
) -> pl.DataFrame:
    """Deterministic GBM daily bars with all required columns."""
    rng = random.Random(seed)
    start = dt.datetime(2024, 1, 2, 9, 30)
    ts, closes, log_rets, atrs, vols, avg_vols, events = [], [], [], [], [], [], []
    price = start_price
    for i in range(n):
        ts.append(start + dt.timedelta(days=i))
        ret = drift + vol * rng.gauss(0, 1)
        price = price * math.exp(ret)
        closes.append(price)
        log_rets.append(ret if i > 0 else None)
        atrs.append(price * 0.015)
        vols.append(max(1, int(1_000_000 * (0.7 + 0.6 * rng.random()))))
        avg_vols.append(900_000.0)
        events.append(False)

    return pl.DataFrame({
        "timestamp":       pl.Series(ts,       dtype=pl.Datetime("us")),
        "close":           pl.Series(closes,   dtype=pl.Float64),
        "log_return":      pl.Series(log_rets, dtype=pl.Float64),
        "atr_14":          pl.Series(atrs,     dtype=pl.Float64),
        "volume":          pl.Series(vols,     dtype=pl.Int64),
        "avg_volume_20":   pl.Series(avg_vols, dtype=pl.Float64),
        "is_event_window": pl.Series(events,   dtype=pl.Boolean),
    })


def _with_signal(df: pl.DataFrame, signal: list[float] | None = None) -> pl.DataFrame:
    sig = signal if signal is not None else [1.0] * len(df)
    return df.with_columns(
        pl.Series("signal", [float(s) for s in sig], dtype=pl.Float64)
    )


def _engine(shares: int = 100, rf: float = 0.05) -> VectorizedEngine:
    return VectorizedEngine(
        initial_capital = 100_000,
        shares_per_unit = shares,
        risk_free_rate  = rf,
        ann_factor      = 252,
    )


# --------------------------------------------------------------------------- #
# Transaction cost tests                                                        #
# --------------------------------------------------------------------------- #

class TestTransactionCosts:
    def test_zero_trade_zero_cost(self):
        c = compute_transaction_costs(
            trade_sizes        = pl.Series([0.0] * 5),
            prices             = pl.Series([100.0] * 5),
            atr                = pl.Series([1.5] * 5),
            liquidity_constant = 1_000_000,
            base_bps           = 5.0,
            impact_coefficient = 50.0,
        )
        assert float(c.sum()) == pytest.approx(0.0)

    def test_cost_positive_for_nonzero_trade(self):
        c = compute_transaction_costs(
            trade_sizes        = pl.Series([500.0]),
            prices             = pl.Series([100.0]),
            atr                = pl.Series([1.5]),
            liquidity_constant = 1_000_000,
            base_bps           = 5.0,
            impact_coefficient = 50.0,
        )
        assert float(c[0]) > 0.0

    def test_event_spike_increases_cost(self):
        """Event window bars should pay higher spread cost."""
        base = compute_transaction_costs(
            trade_sizes        = pl.Series([500.0]),
            prices             = pl.Series([100.0]),
            atr                = pl.Series([1.5]),
            liquidity_constant = 1_000_000,
            base_bps           = 5.0,
            impact_coefficient = 50.0,
            is_event           = pl.Series([False]),
            event_spread_mult  = 5.0,
        )
        event = compute_transaction_costs(
            trade_sizes        = pl.Series([500.0]),
            prices             = pl.Series([100.0]),
            atr                = pl.Series([1.5]),
            liquidity_constant = 1_000_000,
            base_bps           = 5.0,
            impact_coefficient = 50.0,
            is_event           = pl.Series([True]),
            event_spread_mult  = 5.0,
        )
        assert float(event[0]) > float(base[0])

    def test_high_volume_lowers_impact(self):
        """
        High bar volume relative to average → adjusted_lc increases → lower impact.
        """
        low_vol = compute_transaction_costs(
            trade_sizes        = pl.Series([500.0]),
            prices             = pl.Series([100.0]),
            atr                = pl.Series([1.5]),
            liquidity_constant = 1_000_000,
            base_bps           = 5.0,
            impact_coefficient = 50.0,
            bar_volume         = pl.Series([200_000.0]),   # 0.2× avg → illiquid
            avg_bar_volume     = pl.Series([1_000_000.0]),
        )
        high_vol = compute_transaction_costs(
            trade_sizes        = pl.Series([500.0]),
            prices             = pl.Series([100.0]),
            atr                = pl.Series([1.5]),
            liquidity_constant = 1_000_000,
            base_bps           = 5.0,
            impact_coefficient = 50.0,
            bar_volume         = pl.Series([4_000_000.0]),  # 4× avg → liquid
            avg_bar_volume     = pl.Series([1_000_000.0]),
        )
        assert float(high_vol[0]) < float(low_vol[0])

    def test_nvo_more_expensive_than_pfe(self):
        """NVO (smaller LC, higher base bps) should cost more than PFE."""
        pfe = cost_for_ticker("PFE", pl.Series([1000.0]), pl.Series([28.0]), pl.Series([0.4]))
        nvo = cost_for_ticker("NVO", pl.Series([1000.0]), pl.Series([110.0]), pl.Series([1.5]))
        # NVO notional is larger AND higher base_bps → definitely higher cost
        # Normalise by notional so we compare cost per dollar traded
        pfe_bps = float(pfe[0]) / (1000 * 28) * 10_000
        nvo_bps = float(nvo[0]) / (1000 * 110) * 10_000
        assert nvo_bps > pfe_bps

    def test_cost_nonnegative_for_short_trades(self):
        c = compute_transaction_costs(
            trade_sizes        = pl.Series([-500.0, -200.0]),
            prices             = pl.Series([50.0, 50.0]),
            atr                = pl.Series([0.75, 0.75]),
            liquidity_constant = 500_000,
            base_bps           = 5.0,
            impact_coefficient = 50.0,
        )
        assert float(c.min()) >= 0.0


# --------------------------------------------------------------------------- #
# VectorizedEngine tests                                                        #
# --------------------------------------------------------------------------- #

class TestVectorizedEngine:
    def test_always_long_positive_drift_earns_money(self):
        df = _with_signal(_price_df(drift=0.001, vol=0.005))
        r  = _engine().run("PFE", df, "long")
        assert r.total_return > 0.0

    def test_always_short_positive_drift_loses_money(self):
        df = _with_signal(_price_df(drift=0.001, vol=0.005), [-1.0]*252)
        r  = _engine().run("PFE", df, "short")
        assert r.total_return < 0.0

    def test_flat_signal_zero_return_zero_trades(self):
        df = _with_signal(_price_df(), [0.0]*252)
        r  = _engine().run("PFE", df, "flat")
        assert r.total_return == pytest.approx(0.0, abs=1e-9)
        assert r.n_trades == 0

    def test_net_sharpe_below_gross_sharpe(self):
        """Costs must drag net below gross when there are trades."""
        # High-turnover signal: alternate every bar to maximise cost
        signal = [1.0 if i % 2 == 0 else -1.0 for i in range(252)]
        df = _with_signal(_price_df(drift=0.001, vol=0.01), signal)
        r  = _engine(shares=500).run("PFE", df, "churn")
        assert r.net_sharpe < r.gross_sharpe

    def test_mdd_nonpositive(self):
        df = _with_signal(_price_df())
        r  = _engine().run("PFE", df, "test")
        assert r.max_drawdown <= 0.0

    def test_mdd_worse_during_drawdown_period(self):
        """Series with a sharp drop should produce worse MDD than trending up."""
        # Fabricate a series that drops 30% in the middle
        n = 252
        prices = [100.0 * (1 + i * 0.001) for i in range(100)]  # rising
        prices += [prices[-1] * (0.70 ** (1/50)) ** (i+1) for i in range(50)]  # −30% drop
        prices += [prices[-1] * (1 + 0.001) for _ in range(102)]  # recovery
        log_rets = [None] + [math.log(prices[i]/prices[i-1]) for i in range(1, n)]
        atrs = [p * 0.015 for p in prices]
        ts   = [dt.datetime(2024, 1, 2) + dt.timedelta(days=i) for i in range(n)]

        df = pl.DataFrame({
            "timestamp":       pl.Series(ts,       dtype=pl.Datetime("us")),
            "close":           pl.Series(prices,   dtype=pl.Float64),
            "log_return":      pl.Series(log_rets, dtype=pl.Float64),
            "atr_14":          pl.Series(atrs,     dtype=pl.Float64),
            "is_event_window": pl.Series([False]*n, dtype=pl.Boolean),
        }).with_columns(pl.Series("signal", [1.0]*n, dtype=pl.Float64))

        r = VectorizedEngine(
            initial_capital=100_000, shares_per_unit=100, ann_factor=252
        ).run("PFE", df, "drawdown_test")
        assert r.max_drawdown < -0.15   # at least −15% MDD

    def test_equity_starts_near_initial_capital(self):
        df = _with_signal(_price_df(vol=0.01))
        r  = _engine().run("PFE", df, "test")
        first = float(r.equity_curve["equity"][0])
        assert abs(first - 100_000) / 100_000 < 0.02

    def test_missing_required_column_raises(self):
        df = _with_signal(_price_df()).drop("atr_14")
        with pytest.raises(ValueError, match="missing columns"):
            _engine().run("PFE", df, "bad")

    def test_minute_annualisation_factor(self):
        """Engine with ann_factor=98280 should produce different Sharpe than daily."""
        df = _with_signal(_price_df(n=252, drift=0.0001, vol=0.002))
        r_daily  = VectorizedEngine(ann_factor=252,    shares_per_unit=10).run("PFE", df, "d")
        r_minute = VectorizedEngine(ann_factor=98_280, shares_per_unit=10).run("PFE", df, "m")
        # Both use same data, different scaling → Sharpe numerically different
        assert r_daily.net_sharpe != pytest.approx(r_minute.net_sharpe, abs=0.001)

    def test_sortino_gt_sharpe_for_positive_skew_positive_mean(self):
        """
        When mean is positive AND returns are positively skewed, downside_std
        < total_std, so Sortino > Sharpe.

        Note: the inequality reverses when mean < 0 (both metrics negative
        but Sortino becomes more negative because downside_std is smaller).
        We guarantee a positive mean by construction.
        """
        rng = random.Random(7)
        n = 252
        # 80% of bars gain +0.006, 20% of bars lose -0.002.
        # E[r] = 0.8*0.006 + 0.2*(-0.002) = 0.0048 - 0.0004 = 0.0044  (clearly positive)
        log_rets = [
            0.006 if rng.random() > 0.2 else -0.002
            for _ in range(n)
        ]
        prices = [100.0]
        for r in log_rets:
            prices.append(prices[-1] * math.exp(r))

        ts = [dt.datetime(2024, 1, 2) + dt.timedelta(days=i) for i in range(n)]
        df = pl.DataFrame({
            "timestamp":  pl.Series(ts,                    dtype=pl.Datetime("us")),
            "close":      pl.Series(prices[:n],            dtype=pl.Float64),
            "log_return": pl.Series([None]+log_rets[:-1],  dtype=pl.Float64),
            "atr_14":     pl.Series([p*0.01 for p in prices[:n]], dtype=pl.Float64),
        }).with_columns(pl.Series("signal", [1.0]*n, dtype=pl.Float64))

        r = VectorizedEngine(
            initial_capital = 100_000,
            shares_per_unit = 100,
            ann_factor      = 252,
            risk_free_rate  = 0.0,   # zero rf so excess = return (keeps signs clean)
        ).run("PFE", df, "skew_test")

        assert r.net_sharpe > 0, "Mean must be positive for this test to be valid"
        # Downside vol < total vol → Sortino strictly greater than Sharpe
        assert r.sortino > r.net_sharpe


# --------------------------------------------------------------------------- #
# Report / comparison table tests                                               #
# --------------------------------------------------------------------------- #

class TestReport:
    def _two_results(self) -> list[BacktestResult]:
        eng = _engine()
        r1 = eng.run("PFE", _with_signal(_price_df(drift=0.001)),  "momentum")
        r2 = eng.run("NVO", _with_signal(_price_df(drift=0.002)),  "momentum")
        return [r1, r2]

    def test_comparison_table_has_correct_schema(self):
        tbl = build_comparison_table(self._two_results())
        assert "ticker"      in tbl.columns
        assert "net_sharpe"  in tbl.columns
        assert "max_drawdown" in tbl.columns
        assert len(tbl) == 2

    def test_comparison_sorted_by_net_sharpe_desc_per_ticker(self):
        eng = _engine()
        results = [
            eng.run("PFE", _with_signal(_price_df(drift=0.001, vol=0.005)), "mom"),
            eng.run("PFE", _with_signal(_price_df(drift=-0.001, vol=0.02)), "mean_rev"),
        ]
        tbl = build_comparison_table(results)
        pfe_rows = tbl.filter(pl.col("ticker") == "PFE")
        sharpes = pfe_rows["net_sharpe"].to_list()
        assert sharpes == sorted(sharpes, reverse=True)


# --------------------------------------------------------------------------- #
# Event window statistics                                                       #
# --------------------------------------------------------------------------- #

class TestEventWindowStats:
    def test_event_plus_nonevent_returns_sum_to_full_period(self):
        df  = _with_signal(_price_df(n=252, drift=0.001, vol=0.01))
        r   = _engine().run("PFE", df, "test")

        event_mask = pl.Series([i < 20 for i in range(len(r.equity_curve))])
        report = compute_event_window_report(
            ticker          = "PFE",
            strategy_name   = "test",
            equity_curve    = r.equity_curve,
            is_event_window = event_mask,
        )
        full  = report.full_period.total_log_ret
        recon = (report.event_windows.total_log_ret
                 + report.non_event.total_log_ret)
        assert abs(recon - full) < 1e-9

    def test_event_window_sharpe_lower_when_events_are_bad(self):
        """Bars flagged as events should have lower Sharpe if we inject losses there."""
        n = 252
        log_rets = [0.001] * n
        # Inject losses on "event" bars 0–19
        for i in range(20):
            log_rets[i] = -0.05

        ts = [dt.datetime(2024, 1, 2) + dt.timedelta(days=i) for i in range(n)]
        prices = [100.0]
        for lr in log_rets:
            prices.append(prices[-1] * math.exp(lr))

        df = pl.DataFrame({
            "timestamp":  pl.Series(ts,              dtype=pl.Datetime("us")),
            "close":      pl.Series(prices[:n],      dtype=pl.Float64),
            "log_return": pl.Series([None]+log_rets[:-1], dtype=pl.Float64),
            "atr_14":     pl.Series([0.5]*n,         dtype=pl.Float64),
        }).with_columns(pl.Series("signal", [1.0]*n, dtype=pl.Float64))

        r = VectorizedEngine(
            initial_capital=100_000, shares_per_unit=100, ann_factor=252
        ).run("PFE", df, "event_test")

        event_mask = pl.Series([i < 20 for i in range(n)])
        report = compute_event_window_report(
            ticker="PFE", strategy_name="event_test",
            equity_curve=r.equity_curve, is_event_window=event_mask,
        )
        # Event window should drag overall performance
        assert report.event_windows.total_log_ret < report.non_event.total_log_ret
