"""
tests/test_universal.py — Tests for the four universal upgrades:
  1. Dynamic metadata (data/metadata.py)
  2. Cost calibrator (engine/cost_calibrator.py)
  3. Cross-sectional signals (strategy/cross_sectional.py)
"""

from __future__ import annotations

import datetime as dt
import math
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _make_price_df(
    n_bars: int = 200,
    n_tickers: int = 5,
    base_price: float = 50.0,
    median_volume: int = 5_000_000,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Create a synthetic long-format OHLCV DataFrame for testing.

    Returns columns: ticker, timestamp, open, high, low, close, volume, atr_14
    """
    import random
    rng = random.Random(seed)
    tickers = [f"T{i}" for i in range(n_tickers)]

    timestamps = [
        dt.datetime(2023, 1, 1) + dt.timedelta(days=i)
        for i in range(n_bars)
    ]

    rows = []
    for ticker in tickers:
        price = base_price
        for ts in timestamps:
            ret = rng.gauss(0.0, 0.02)
            close = price * math.exp(ret)
            high = close * (1.0 + abs(rng.gauss(0, 0.005)))
            low = close * (1.0 - abs(rng.gauss(0, 0.005)))
            vol = int(median_volume * (0.5 + rng.random()))
            rows.append({
                "ticker": ticker,
                "timestamp": ts,
                "open": price,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
            })
            price = close

    df = pl.DataFrame(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us")),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
    )

    # Add atr_14 using the high-low range EWM
    prev_c = pl.col("close").shift(1).over("ticker")
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_c).abs(),
        (pl.col("low") - prev_c).abs(),
    )
    df = df.with_columns(tr.alias("_tr")).with_columns(
        pl.col("_tr").ewm_mean(span=14, adjust=False).over("ticker").alias("atr_14")
    ).drop("_tr")

    return df


def _make_single_ticker_df(
    n_bars: int = 200,
    median_volume: int = 5_000_000,
    base_price: float = 50.0,
    seed: int = 0,
) -> pl.DataFrame:
    """Single-ticker DataFrame (no 'ticker' column) with atr_14."""
    long_df = _make_price_df(
        n_bars=n_bars, n_tickers=1,
        median_volume=median_volume,
        base_price=base_price,
        seed=seed,
    )
    return long_df.filter(pl.col("ticker") == "T0").drop("ticker").sort("timestamp")


# ============================================================================ #
# TestDynamicMetadata                                                           #
# ============================================================================ #

class TestDynamicMetadata:

    def test_static_tickers_use_hardcoded_meta(self):
        """PFE and NVO return from STATIC_META without any network call."""
        from backtester.data.metadata import get_metadata, STATIC_META

        with patch("backtester.data.metadata.fetch_live_metadata") as mock_fetch:
            pfe_meta = get_metadata("PFE")
            nvo_meta = get_metadata("NVO")

        # fetch_live_metadata must NOT have been called
        mock_fetch.assert_not_called()

        # Must match STATIC_META exactly
        assert pfe_meta == STATIC_META["PFE"]
        assert nvo_meta == STATIC_META["NVO"]

        # Spot-check key fields
        assert pfe_meta["avg_daily_volume"] == 28_000_000
        assert nvo_meta["exchange"] == "NYSE"

    def test_unknown_ticker_fetches_live(self, tmp_path):
        """Unknown ticker calls fetch_live_metadata and returns expected keys."""
        from backtester.data import metadata as meta_mod

        fake_meta = {
            "name": "Apple Inc.",
            "exchange": "NASDAQ",
            "avg_daily_volume": 80_000_000,
            "approx_price": 175.0,
            "approx_price_2024": 175.0,
            "market_cap": 2_800_000_000_000.0,
            "sector": "Technology",
        }

        # Redirect cache to tmp_path so we don't pollute the real cache
        original_cache = meta_mod.CACHE_PATH
        meta_mod.CACHE_PATH = tmp_path / "metadata.json"

        try:
            with patch.object(meta_mod, "fetch_live_metadata", return_value=fake_meta) as mock_fetch:
                result = meta_mod.get_metadata("AAPL", use_cache=True)

            mock_fetch.assert_called_once_with("AAPL")
        finally:
            meta_mod.CACHE_PATH = original_cache

        # All expected keys present
        for key in ("name", "exchange", "avg_daily_volume", "approx_price", "market_cap", "sector"):
            assert key in result, f"Missing key: {key}"

        assert result["exchange"] == "NASDAQ"
        assert result["avg_daily_volume"] == 80_000_000

    def test_get_metadata_batch(self, tmp_path):
        """get_metadata_batch returns a dict keyed by all requested tickers."""
        from backtester.data import metadata as meta_mod

        fake_meta = {
            "name": "Generic",
            "exchange": "NYSE",
            "avg_daily_volume": 1_000_000,
            "approx_price": 50.0,
            "approx_price_2024": 50.0,
            "market_cap": None,
            "sector": "Unknown",
        }

        original_cache = meta_mod.CACHE_PATH
        meta_mod.CACHE_PATH = tmp_path / "metadata.json"

        try:
            # PFE/NVO come from STATIC_META; TSLA is unknown → mocked
            with patch.object(meta_mod, "fetch_live_metadata", return_value=fake_meta):
                result = meta_mod.get_metadata_batch(["PFE", "NVO", "TSLA"], max_workers=2)
        finally:
            meta_mod.CACHE_PATH = original_cache

        assert set(result.keys()) == {"PFE", "NVO", "TSLA"}
        # PFE/NVO come from STATIC_META unchanged
        assert result["PFE"]["avg_daily_volume"] == 28_000_000
        # TSLA got the mocked meta
        assert result["TSLA"]["exchange"] == "NYSE"


# ============================================================================ #
# TestCostCalibrator                                                            #
# ============================================================================ #

class TestCostCalibrator:

    def test_calibrate_from_synthetic_df(self):
        """Calibrated params must stay within valid formula bounds."""
        from backtester.engine.cost_calibrator import calibrate_from_df

        df = _make_single_ticker_df(n_bars=200, median_volume=5_000_000)
        params = calibrate_from_df("SYNTH", df)

        assert 2.0 <= params["base_bps"] <= 20.0
        assert 5.0 <= params["impact_coefficient"] <= 80.0
        assert params["event_spread_multiplier"] == 4.0
        assert params["base_liquidity"] > 0

    def test_known_ticker_uses_cost_params(self):
        """PFE must return the hardcoded COST_PARAMS entry unchanged."""
        from backtester.engine.cost_calibrator import get_cost_params
        from backtester.engine.costs import COST_PARAMS

        df = _make_single_ticker_df()
        result = get_cost_params("PFE", df)

        assert result is COST_PARAMS["PFE"]
        assert result["base_bps"] == 5.0
        assert result["impact_coefficient"] == 16.0

    def test_unknown_ticker_calibrates_from_data(self):
        """Ticker not in COST_PARAMS must be calibrated from the supplied df."""
        from backtester.engine.cost_calibrator import get_cost_params
        from backtester.engine.costs import COST_PARAMS

        assert "AAPL" not in COST_PARAMS, "Test assumes AAPL is not in COST_PARAMS"

        df = _make_single_ticker_df()
        params = get_cost_params("AAPL", df)

        assert "base_bps" in params
        assert "impact_coefficient" in params
        assert "base_liquidity" in params
        assert 2.0 <= params["base_bps"] <= 20.0

    def test_high_volume_gives_low_impact(self):
        """
        High-volume ticker (80M shares/day) should have lower impact_coefficient
        than a low-volume ticker.  The formula gives κ = 300_000 / sqrt(V), so
        higher V → lower κ.  The absolute floor (5.0) is only reached when
        V ≥ (300_000/5)^2 = 3.6 billion shares/day.

        Test: high-volume κ < low-volume κ, and both are in [5, 80].
        Additionally test the absolute floor with a synthetic extreme volume.
        """
        from backtester.engine.cost_calibrator import calibrate_from_df

        df_high = _make_single_ticker_df(median_volume=80_000_000)
        df_low  = _make_single_ticker_df(median_volume=200_000)

        params_high = calibrate_from_df("HIGH_VOL", df_high)
        params_low  = calibrate_from_df("LOW_VOL", df_low)

        # High volume must produce lower κ
        assert params_high["impact_coefficient"] < params_low["impact_coefficient"], (
            f"High-volume ticker ({params_high['impact_coefficient']:.2f}) should have "
            f"lower κ than low-volume ({params_low['impact_coefficient']:.2f})"
        )

        # Both must be within formula bounds
        assert 5.0 <= params_high["impact_coefficient"] <= 80.0
        assert 5.0 <= params_low["impact_coefficient"] <= 80.0

        # Verify floor: an extreme volume (e.g., 4B) must clip to exactly 5.0
        # Build a minimal df with a very high volume column directly
        import polars as pl
        df_extreme = pl.DataFrame({
            "close":  pl.Series([50.0] * 50, dtype=pl.Float64),
            "volume": pl.Series([4_000_000_000] * 50, dtype=pl.Int64),
            "atr_14": pl.Series([1.0] * 50, dtype=pl.Float64),
        })
        params_extreme = calibrate_from_df("EXTREME", df_extreme)
        assert params_extreme["impact_coefficient"] == 5.0, (
            f"Expected floor 5.0 for extreme volume, got {params_extreme['impact_coefficient']}"
        )

    def test_low_volume_gives_high_impact(self):
        """100k share/day ticker should produce impact_coefficient close to ceiling (80.0)."""
        from backtester.engine.cost_calibrator import calibrate_from_df

        # Very low volume → κ = 300_000 / sqrt(100_000) ≈ 948 → clipped to 80.0
        df = _make_single_ticker_df(median_volume=100_000)
        params = calibrate_from_df("MICROCAP", df)

        assert params["impact_coefficient"] >= 75.0, (
            f"Expected impact_coefficient near 80.0 for low-volume ticker, got {params['impact_coefficient']}"
        )


# ============================================================================ #
# TestCrossSectionalSignals                                                     #
# ============================================================================ #

class TestCrossSectionalSignals:

    def test_momentum_rank_signal_shape(self):
        """Output must have same row count as input and correct columns."""
        from backtester.strategy.cross_sectional import momentum_rank_signal

        long_df = _make_price_df(n_bars=200, n_tickers=5)
        result = momentum_rank_signal(long_df)

        assert len(result) == len(long_df)
        assert set(result.columns) == {"ticker", "timestamp", "signal"}

    def test_momentum_rank_signal_values(self):
        """Signal values must be exactly in {-1.0, 0.0, +1.0}."""
        from backtester.strategy.cross_sectional import momentum_rank_signal

        long_df = _make_price_df(n_bars=200, n_tickers=5)
        result = momentum_rank_signal(long_df)

        valid_values = {-1.0, 0.0, 1.0}
        unique_signals = set(result["signal"].to_list())
        # null/nan can appear in warmup bars — cast those out
        unique_non_null = {v for v in unique_signals if v is not None and not (isinstance(v, float) and math.isnan(v))}
        assert unique_non_null.issubset(valid_values), (
            f"Signal contains unexpected values: {unique_non_null - valid_values}"
        )

    def test_rank_signal_market_neutral(self):
        """
        Long-short portfolio: for timestamps where BOTH longs AND shorts are
        present (i.e., a fully active bar), the sum of signals should be 0.

        After the PiT shift, warmup bars and transition bars can have partial
        fills (only longs or only shorts from the previous bar's signal
        propagating forward). We test market-neutrality only on bars that have
        at least one long AND at least one short simultaneously.
        """
        from backtester.strategy.cross_sectional import momentum_rank_signal

        # Use 10 tickers for cleaner long-short balance
        long_df = _make_price_df(n_bars=200, n_tickers=10, seed=99)
        result = momentum_rank_signal(long_df, top_pct=0.2, bottom_pct=0.2)

        # Identify fully active bars: those with at least one +1 AND at least one -1
        per_ts_counts = (
            result
            .group_by("timestamp")
            .agg([
                (pl.col("signal") == 1.0).sum().alias("n_long"),
                (pl.col("signal") == -1.0).sum().alias("n_short"),
                pl.col("signal").sum().alias("net_signal"),
            ])
        )
        # Filter to bars that have both longs and shorts present
        balanced_bars = per_ts_counts.filter(
            (pl.col("n_long") > 0) & (pl.col("n_short") > 0)
        )

        if len(balanced_bars) == 0:
            pytest.skip("No fully balanced signal bars found (may need more bars)")

        # On balanced bars, longs and shorts should cancel out (sum = 0)
        max_imbalance = float(balanced_bars["net_signal"].abs().max() or 0.0)
        assert max_imbalance < 1e-9, (
            f"Signal sum not market-neutral on balanced bars; max |net| = {max_imbalance}"
        )

    def test_mean_reversion_rank_signal_shape(self):
        """Mean reversion signal output must match input shape and columns."""
        from backtester.strategy.cross_sectional import mean_reversion_rank_signal

        long_df = _make_price_df(n_bars=200, n_tickers=5)
        result = mean_reversion_rank_signal(long_df)

        assert len(result) == len(long_df)
        assert set(result.columns) == {"ticker", "timestamp", "signal"}

    def test_pit_shift_applied(self):
        """
        First bar's signal for every ticker must be 0 — no look-ahead on bar 0.

        After the PiT shift(1), bar 0 must carry a null/0 signal regardless of
        what the raw signal would have been.
        """
        from backtester.strategy.cross_sectional import momentum_rank_signal

        long_df = _make_price_df(n_bars=200, n_tickers=5)
        result = momentum_rank_signal(long_df)

        # Get the earliest timestamp overall
        min_ts = long_df["timestamp"].min()

        first_bar_signals = (
            result
            .filter(pl.col("timestamp") == min_ts)
            ["signal"]
            .to_list()
        )

        for sig in first_bar_signals:
            assert sig == 0.0 or sig is None, (
                f"First bar signal should be 0 (PiT), got {sig}"
            )
