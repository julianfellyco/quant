"""
tests/test_universe.py — Unit tests for universe_fetcher, screener, and universe_engine.

All yfinance network calls are mocked via unittest.mock.patch so tests run
fully offline.
"""

from __future__ import annotations

import datetime as dt
import math
from unittest.mock import MagicMock, patch

import polars as pl
import pytest


# --------------------------------------------------------------------------- #
# Helpers: synthetic data builders                                              #
# --------------------------------------------------------------------------- #

def _make_pandas_ohlcv(n: int = 50, ticker: str = "TEST") -> "pd.DataFrame":
    """Build a minimal pandas OHLCV DataFrame mimicking yfinance Ticker.history()."""
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-03", periods=n, freq="B")
    close = 100.0 + rng.normal(0, 1, n).cumsum()
    df = pd.DataFrame(
        {
            "Open":   close * 0.999,
            "High":   close * 1.005,
            "Low":    close * 0.995,
            "Close":  close,
            "Volume": rng.integers(500_000, 5_000_000, n),
        },
        index=dates,
    )
    df.index.name = "Datetime"
    return df


def _make_long_df(tickers: list[str], n_bars: int = 100) -> pl.DataFrame:
    """
    Build a synthetic long-format Polars DataFrame with required columns.
    Includes log_return, atr_14, and zscore_20d so signal functions work.
    """
    import numpy as np

    frames = []
    rng = np.random.default_rng(0)

    for i, ticker in enumerate(tickers):
        dates = [
            dt.datetime(2023, 1, 3) + dt.timedelta(days=d)
            for d in range(n_bars)
        ]
        close = 50.0 + rng.normal(0, 0.5, n_bars).cumsum() + i * 10
        close = [max(c, 1.0) for c in close]

        df = pl.DataFrame({
            "ticker":    [ticker] * n_bars,
            "timestamp": dates,
            "open":      [c * 0.999 for c in close],
            "high":      [c * 1.005 for c in close],
            "low":       [c * 0.995 for c in close],
            "close":     close,
            "volume":    [int(v) for v in rng.integers(1_000_000, 5_000_000, n_bars)],
        }).with_columns([
            pl.col("timestamp").cast(pl.Datetime("us")),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
        ])

        # Add derived columns needed by engine
        log_ret = (pl.col("close") / pl.col("close").shift(1)).log()
        df = df.with_columns(log_ret.alias("log_return"))
        df = df.with_columns(
            (pl.col("high") - pl.col("low"))
            .ewm_mean(span=14, adjust=False)
            .alias("atr_14")
        )

        frames.append(df)

    return pl.concat(frames, how="diagonal").sort(["ticker", "timestamp"])


# --------------------------------------------------------------------------- #
# test_fetch_universe_long_shape                                                #
# --------------------------------------------------------------------------- #

class TestFetchUniverseLong:
    """Tests for data.universe_fetcher.fetch_universe_long."""

    @patch("yfinance.Ticker")
    def test_fetch_universe_long_shape(self, mock_ticker_cls):
        """
        fetch_universe_long should return a DataFrame with a 'ticker' column
        and roughly n_tickers × n_bars rows.
        """
        import pandas as pd

        tickers = ["AAPL", "MSFT", "GOOG"]
        n_bars  = 50

        # Configure mock: every Ticker().history() returns the same synthetic frame
        mock_instance = MagicMock()
        mock_instance.history.return_value = _make_pandas_ohlcv(n=n_bars)
        mock_ticker_cls.return_value = mock_instance

        from backtester.data.universe_fetcher import fetch_universe_long

        long_df = fetch_universe_long(
            tickers     = tickers,
            start       = "2023-01-01",
            end         = "2023-12-31",
            granularity = "daily",
            max_workers = 3,
            force       = True,   # skip cache check
        )

        assert "ticker" in long_df.columns, "Expected 'ticker' column in long DataFrame"
        assert len(long_df) > 0, "Long DataFrame should not be empty"

        # Check that each requested ticker appears
        found_tickers = set(long_df["ticker"].unique().to_list())
        for t in tickers:
            assert t in found_tickers, f"Ticker {t} missing from long DataFrame"

        # Row count: each ticker should have bars in the filtered window.
        # The synthetic frame has 50 bars; after date filtering some may be excluded
        # so just confirm we have at least 1 row per ticker.
        per_ticker = long_df.group_by("ticker").agg(pl.len().alias("cnt"))
        for row in per_ticker.iter_rows(named=True):
            assert row["cnt"] >= 1, f"{row['ticker']} has no rows"

    @patch("yfinance.Ticker")
    def test_fetch_universe_long_ticker_column_is_string(self, mock_ticker_cls):
        """The 'ticker' column dtype must be Utf8 (string)."""
        mock_instance = MagicMock()
        mock_instance.history.return_value = _make_pandas_ohlcv(n=20)
        mock_ticker_cls.return_value = mock_instance

        from backtester.data.universe_fetcher import fetch_universe_long

        long_df = fetch_universe_long(
            tickers=["TSLA"],
            start="2023-01-01",
            end="2023-12-31",
            force=True,
        )

        assert long_df["ticker"].dtype == pl.Utf8, (
            f"Expected Utf8 dtype for ticker column, got {long_df['ticker'].dtype}"
        )


# --------------------------------------------------------------------------- #
# test_screener_pharma_large_cap                                                #
# --------------------------------------------------------------------------- #

class TestScreenerPharmaLargeCap:
    """Tests for data.screener.Screener.pharma_large_cap."""

    def test_returns_list_of_strings(self):
        from backtester.data.screener import Screener

        result = Screener.pharma_large_cap()
        assert isinstance(result, list), "pharma_large_cap() should return a list"
        assert all(isinstance(t, str) for t in result), "All tickers should be strings"

    def test_minimum_length(self):
        from backtester.data.screener import Screener

        result = Screener.pharma_large_cap()
        assert len(result) >= 10, (
            f"Expected at least 10 pharma tickers, got {len(result)}"
        )

    def test_all_uppercase(self):
        from backtester.data.screener import Screener

        result = Screener.pharma_large_cap()
        for ticker in result:
            assert ticker == ticker.upper(), (
                f"Ticker {ticker!r} is not uppercase"
            )

    def test_contains_known_tickers(self):
        from backtester.data.screener import Screener

        result = Screener.pharma_large_cap()
        # A few anchors that should always be present
        for expected in ("PFE", "NVO", "LLY", "JNJ"):
            assert expected in result, f"Expected {expected} in pharma_large_cap()"


# --------------------------------------------------------------------------- #
# test_screener_by_volume_filter                                                #
# --------------------------------------------------------------------------- #

class TestScreenerByVolumeFilter:
    """Tests for data.screener.Screener.by_volume_filter."""

    def _make_yf_download_mock(self, volumes: dict[str, list[float]]) -> MagicMock:
        """
        Build a mock return value for yfinance.download() that returns a
        MultiIndex DataFrame where data["Volume"][ticker] works.
        """
        import pandas as pd
        import numpy as np

        tickers = list(volumes.keys())
        n       = max(len(v) for v in volumes.values())
        dates   = pd.date_range("2023-11-01", periods=n, freq="B")

        # Build MultiIndex columns: (Price, Ticker)
        tuples = [("Volume", t) for t in tickers]
        midx   = pd.MultiIndex.from_tuples(tuples)

        data = pd.DataFrame(
            {(price, t): volumes[t] for price, t in tuples},
            index=dates,
        )
        data.columns = midx
        return data

    @patch("yfinance.download")
    def test_filters_below_threshold(self, mock_download):
        """Tickers below min_avg_volume must be excluded."""
        import pandas as pd

        # AAPL: avg volume 3M (passes). TINY: avg volume 200K (fails).
        volumes = {
            "AAPL": [3_000_000] * 25,
            "TINY": [200_000]   * 25,
        }
        mock_download.return_value = self._make_yf_download_mock(volumes)

        from backtester.data.screener import Screener

        result = Screener.by_volume_filter(
            tickers        = ["AAPL", "TINY"],
            min_avg_volume = 1_000_000,
            lookback_days  = 20,
        )

        assert "AAPL" in result,  "AAPL (3M avg vol) should pass the filter"
        assert "TINY" not in result, "TINY (200K avg vol) should fail the filter"

    @patch("yfinance.download")
    def test_all_pass_when_high_volume(self, mock_download):
        """All tickers should pass when all average volumes exceed threshold."""
        volumes = {
            "A": [5_000_000] * 25,
            "B": [2_000_000] * 25,
            "C": [8_000_000] * 25,
        }
        mock_download.return_value = self._make_yf_download_mock(volumes)

        from backtester.data.screener import Screener

        result = Screener.by_volume_filter(
            tickers        = ["A", "B", "C"],
            min_avg_volume = 1_000_000,
            lookback_days  = 20,
        )

        assert set(result) == {"A", "B", "C"}, (
            f"Expected all 3 tickers to pass, got {result}"
        )

    @patch("yfinance.download")
    def test_empty_result_when_none_pass(self, mock_download):
        """Empty list when all tickers fall below threshold."""
        volumes = {
            "X": [50_000] * 25,
            "Y": [30_000] * 25,
        }
        mock_download.return_value = self._make_yf_download_mock(volumes)

        from backtester.data.screener import Screener

        result = Screener.by_volume_filter(
            tickers        = ["X", "Y"],
            min_avg_volume = 1_000_000,
            lookback_days  = 20,
        )

        assert result == [], f"Expected empty list, got {result}"


# --------------------------------------------------------------------------- #
# test_universe_engine_run                                                      #
# --------------------------------------------------------------------------- #

class TestUniverseEngineRun:
    """Tests for engine.universe_engine.UniverseEngine.run."""

    @pytest.fixture
    def long_df(self):
        return _make_long_df(["AAA", "BBB", "CCC"], n_bars=100)

    @staticmethod
    def trivial_signal_fn(df: pl.DataFrame) -> pl.Series:
        """Trivial always-long signal: returns 1.0 for every bar."""
        return pl.Series("signal", [1.0] * len(df), dtype=pl.Float64)

    def test_result_has_all_tickers(self, long_df):
        from backtester.engine.universe_engine import UniverseEngine

        engine = UniverseEngine(initial_capital=100_000, shares_per_unit=100)
        result = engine.run(long_df, self.trivial_signal_fn, strategy_name="always_long")

        assert set(result.results.keys()) == {"AAA", "BBB", "CCC"}, (
            f"Expected 3 tickers in results, got {set(result.results.keys())}"
        )

    def test_net_sharpe_values_are_finite(self, long_df):
        from backtester.engine.universe_engine import UniverseEngine

        engine = UniverseEngine(initial_capital=100_000, shares_per_unit=100)
        result = engine.run(long_df, self.trivial_signal_fn, strategy_name="always_long")

        for ticker, br in result.results.items():
            assert math.isfinite(br.net_sharpe), (
                f"net_sharpe for {ticker} is not finite: {br.net_sharpe}"
            )

    def test_universe_metrics_shape(self, long_df):
        """universe_metrics should have one row per ticker."""
        from backtester.engine.universe_engine import UniverseEngine

        engine = UniverseEngine(initial_capital=100_000, shares_per_unit=100)
        result = engine.run(long_df, self.trivial_signal_fn)

        assert len(result.universe_metrics) == 3, (
            f"Expected 3 rows in universe_metrics, got {len(result.universe_metrics)}"
        )

        expected_cols = {
            "ticker", "strategy", "net_sharpe", "sortino",
            "total_return", "max_drawdown", "n_trades", "total_cost_usd",
        }
        actual_cols = set(result.universe_metrics.columns)
        assert expected_cols.issubset(actual_cols), (
            f"Missing columns: {expected_cols - actual_cols}"
        )

    def test_n_trades_positive(self, long_df):
        """With always-long signal, there should be at least 1 trade per ticker."""
        from backtester.engine.universe_engine import UniverseEngine

        engine = UniverseEngine(initial_capital=100_000, shares_per_unit=100)
        result = engine.run(long_df, self.trivial_signal_fn)

        for ticker, br in result.results.items():
            assert br.n_trades >= 1, f"{ticker} should have at least 1 trade"

    def test_equity_curve_present(self, long_df):
        """Each BacktestResult should have a non-empty equity_curve DataFrame."""
        from backtester.engine.universe_engine import UniverseEngine

        engine = UniverseEngine(initial_capital=100_000, shares_per_unit=100)
        result = engine.run(long_df, self.trivial_signal_fn)

        for ticker, br in result.results.items():
            assert len(br.equity_curve) > 0, (
                f"{ticker} equity_curve is empty"
            )
            assert "equity" in br.equity_curve.columns, (
                f"{ticker} equity_curve missing 'equity' column"
            )


# --------------------------------------------------------------------------- #
# test_universe_engine_top_by_sharpe                                            #
# --------------------------------------------------------------------------- #

class TestUniverseEngineTopBySharpe:
    """Tests for UniverseResult.top_by_sharpe."""

    @pytest.fixture
    def universe_result(self):
        from backtester.engine.universe_engine import UniverseEngine

        long_df = _make_long_df(["T1", "T2", "T3", "T4", "T5"], n_bars=120)

        engine = UniverseEngine(initial_capital=100_000, shares_per_unit=100)
        return engine.run(
            long_df,
            signal_fn=lambda df: pl.Series("signal", [1.0] * len(df), dtype=pl.Float64),
            strategy_name="test_strategy",
        )

    def test_top_by_sharpe_returns_n_rows(self, universe_result):
        top2 = universe_result.top_by_sharpe(n=2)
        assert len(top2) == 2, f"Expected 2 rows, got {len(top2)}"

    def test_top_by_sharpe_sorted_descending(self, universe_result):
        """Rows must be sorted by net_sharpe descending."""
        top = universe_result.top_by_sharpe(n=5)
        sharpes = top["net_sharpe"].to_list()
        assert sharpes == sorted(sharpes, reverse=True), (
            f"Sharpe values not sorted descending: {sharpes}"
        )

    def test_top_by_sharpe_has_ticker_column(self, universe_result):
        top = universe_result.top_by_sharpe(n=3)
        assert "ticker" in top.columns, "top_by_sharpe result missing 'ticker' column"

    def test_top_by_sharpe_n_larger_than_universe(self, universe_result):
        """Requesting more rows than universe size should return all rows."""
        all_rows = universe_result.top_by_sharpe(n=999)
        assert len(all_rows) == len(universe_result.universe_metrics), (
            "top_by_sharpe(n=999) should return all rows when n > universe size"
        )

    def test_summary_returns_string(self, universe_result):
        summary = universe_result.summary()
        assert isinstance(summary, str), "summary() should return a string"
        assert len(summary) > 0, "summary() should not be empty"
