"""
data/handler.py — DataHandler: Polars-backed OHLCV ingestion for daily and
                  1-minute bars, with pharma-specific feature engineering.

Design goals
------------
1. Parquet-first: read from .parquet files for efficient columnar access.
   Falls back to yfinance download + cache if no file is present.
2. Granularity-aware: daily bars use ann_factor=252; 1-minute bars use
   ann_factor=252×390=98,280.  All downstream stats pick this up automatically.
3. Volume as a first-class feature: avg_bar_volume (20-bar rolling mean) is
   always computed so the cost model can perform volume-based liquidity adjustment.
4. Binary event mask: bars inside a pre/post event window are flagged so the
   cost model can spike spreads and the stats module can isolate event P&L.

Parquet schema expected (1-minute)
-----------------------------------
    timestamp   Datetime[us] or Date
    open        Float64
    high        Float64
    low         Float64
    close       Float64
    volume      Int64

The handler adds derived columns and never mutates the source file.

Memory note
-----------
pl.scan_parquet() is used for the initial load.  This builds a lazy query plan:
  • Only requested columns are decoded from disk.
  • Filter pushdown: date-range filters are applied at the Parquet row-group
    level, so a 500M-row tick file for a 60-day window reads only the matching
    row groups.
  • .collect() materialises into memory only after filters/selects are applied.
"""

from __future__ import annotations

import datetime as dt
import math
import random
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

from .events import build_event_mask, get_event_dates

DATA_DIR = Path(__file__).parent.parent / "data_cache"
DATA_DIR.mkdir(exist_ok=True)

TICKER_META: dict[str, dict] = {
    "PFE": {
        "name": "Pfizer Inc.",
        "exchange": "NYSE",
        "avg_daily_volume": 28_000_000,
        "approx_price_2024": 28.0,
    },
    "NVO": {
        "name": "Novo Nordisk ADR",
        "exchange": "NYSE",
        "avg_daily_volume": 3_500_000,
        "approx_price_2024": 110.0,
    },
}

MINUTES_PER_DAY    = 390          # 9:30–16:00 NYSE
HOURS_PER_DAY      = 7            # hourly bars: 9:30,10:30,…,15:30
TRADING_DAYS       = 252
ANN_FACTOR_DAILY   = TRADING_DAYS
ANN_FACTOR_HOUR    = TRADING_DAYS * HOURS_PER_DAY     # 1,764
ANN_FACTOR_MINUTE  = TRADING_DAYS * MINUTES_PER_DAY   # 98,280


class Granularity(Enum):
    DAILY  = auto()
    HOUR   = auto()   # matches data_cache/{ticker}_hour.parquet from downloader.py
    MINUTE = auto()


# --------------------------------------------------------------------------- #
# DataHandler                                                                   #
# --------------------------------------------------------------------------- #

class DataHandler:
    """
    Unified data access for one or more tickers at daily or 1-minute resolution.

    Usage — from parquet files (production path)
    --------------------------------------------
    Place pre-downloaded parquet files at:
        data_cache/PFE_1min.parquet   (1-minute bars)
        data_cache/NVO_1min.parquet
    Then:
        handler = DataHandler(["PFE", "NVO"], granularity=Granularity.MINUTE)
        handler.load(start=date(2024,1,1), end=date(2025,1,1))

    Usage — synthetic data (dev / CI path)
    ---------------------------------------
    If no parquet file is found the handler generates a realistic synthetic
    series via GBM with intraday volume and volatility patterns.  This lets
    all downstream code run without a data vendor subscription.

    Point-in-Time guarantee
    -----------------------
    All indicator columns (rolling vol, z-score, momentum) use .shift(1)
    so bar t only sees information from bars ≤ t−1.
    align_signals() adds one further shift so signals execute on bar t+1.
    """

    def __init__(
        self,
        tickers:     List[str],
        granularity: Granularity = Granularity.DAILY,
    ) -> None:
        from .metadata import get_metadata
        # Accept any ticker — unknown ones get live-fetched metadata
        self._meta = {t: get_metadata(t) for t in tickers}
        self._tickers     = tickers
        self._granularity = granularity
        self._frames:  Dict[str, pl.DataFrame] = {}
        self._paths:   Dict[str, Path]         = {}

    # ------------------------------------------------------------------ #
    # Public: load                                                          #
    # ------------------------------------------------------------------ #

    def load(
        self,
        start: dt.date,
        end:   dt.date,
        force: bool = False,
    ) -> "DataHandler":
        for ticker in self._tickers:
            raw = self._load_raw(ticker, start, end, force)
            df  = self._engineer_features(raw, ticker, start, end)
            self._frames[ticker] = df.sort("timestamp")

            featured_path = DATA_DIR / f"{ticker}_{self._granularity.name.lower()}_featured.parquet"
            df.write_parquet(featured_path)
            self._paths[ticker] = featured_path

        return self

    # ------------------------------------------------------------------ #
    # Public: accessors                                                     #
    # ------------------------------------------------------------------ #

    def __getitem__(self, ticker: str) -> pl.DataFrame:
        if ticker not in self._frames:
            raise KeyError(f"'{ticker}' not loaded. Call .load() first.")
        return self._frames[ticker]

    def scan(self, ticker: str) -> pl.LazyFrame:
        """
        Lazy frame for memory-mapped access.  Prefer this for large 1m datasets.

        Example — read only closing prices for Q1 2024:
            lf = handler.scan("NVO")
            q1 = lf.filter(pl.col("timestamp").dt.year() == 2024,
                            pl.col("timestamp").dt.month() <= 3)
                    .select(["timestamp", "close", "log_return"])
                    .collect()
        """
        if ticker not in self._paths:
            raise KeyError(f"'{ticker}' not loaded.")
        return pl.scan_parquet(self._paths[ticker])

    def align_signals(self, ticker: str, signals: pl.Series) -> pl.DataFrame:
        """
        Attach a signal series and enforce PiT execution lag.

        Signals are shifted forward by 1 bar: a signal generated at bar t
        is *executed* at bar t+1.  This is the single enforcement point for
        look-ahead prevention.
        """
        df = self._frames[ticker]
        if len(signals) != len(df):
            raise ValueError(
                f"Signal length {len(signals)} ≠ price frame length {len(df)}"
            )
        return df.with_columns(
            signals.shift(1).fill_null(0.0).alias("signal")
        )

    @property
    def ann_factor(self) -> int:
        return {
            Granularity.DAILY:  ANN_FACTOR_DAILY,
            Granularity.HOUR:   ANN_FACTOR_HOUR,
            Granularity.MINUTE: ANN_FACTOR_MINUTE,
        }[self._granularity]

    def liquidity_constant(self, ticker: str) -> int:
        """Per-bar average volume, used as the baseline liquidity_constant."""
        base_adv = self._meta[ticker]["avg_daily_volume"]
        return {
            Granularity.DAILY:  base_adv,
            Granularity.HOUR:   base_adv // HOURS_PER_DAY,
            Granularity.MINUTE: base_adv // MINUTES_PER_DAY,
        }[self._granularity]

    # ------------------------------------------------------------------ #
    # Private: raw data loading                                             #
    # ------------------------------------------------------------------ #

    def _load_raw(
        self,
        ticker: str,
        start:  dt.date,
        end:    dt.date,
        force:  bool,
    ) -> pl.DataFrame:
        suffix = {
            Granularity.DAILY:  "daily",
            Granularity.HOUR:   "hour",    # written by downloader.py
            Granularity.MINUTE: "1min",
        }[self._granularity]
        parquet_path = DATA_DIR / f"{ticker}_{suffix}.parquet"

        if parquet_path.exists() and not force:
            print(f"  [{ticker}] Reading from {parquet_path.name}")
            df = (
                pl.scan_parquet(parquet_path)
                .filter(
                    pl.col("timestamp").cast(pl.Date).is_between(
                        pl.lit(start), pl.lit(end)
                    )
                )
                .collect()
            )
            if len(df) > 0:
                return self._normalise_schema(df)

        # No parquet file found — try yfinance then fall back to synthetic
        return self._fetch_or_synthesise(ticker, start, end)

    def _fetch_or_synthesise(
        self,
        ticker: str,
        start:  dt.date,
        end:    dt.date,
    ) -> pl.DataFrame:
        if self._granularity is Granularity.DAILY:
            return self._fetch_daily_yf(ticker, start, end)

        if self._granularity is Granularity.HOUR:
            # Hourly data: yfinance supports up to 730 days back.
            # Direct download works for a 1-year window; use downloader.py for
            # controlled caching (run `python -m backtester.data.downloader` once).
            try:
                from .downloader import download_pharma_data
                frames = download_pharma_data([ticker], period="1y", interval="1h")
                if ticker in frames and len(frames[ticker]) > 100:
                    return frames[ticker]
            except Exception as exc:
                print(f"  [{ticker}] Hourly yfinance fetch failed ({exc}); "
                      f"falling back to synthetic 1-minute data resampled to hourly.")
            return self._synthesise_minute(ticker, start, end)

        # MINUTE: yfinance only provides 1m data for the last ~7 days.
        try:
            df = self._fetch_minute_yf(ticker, start, end)
            if len(df) > 1000:
                return df
        except Exception:
            pass
        print(f"  [{ticker}] Generating synthetic 1-minute data ({start} → {end})")
        return self._synthesise_minute(ticker, start, end)

    # ------------------------------------------------------------------ #
    # yfinance fetchers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fetch_daily_yf(ticker: str, start: dt.date, end: dt.date) -> pl.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("pip install yfinance") from exc

        raw = yf.download(ticker, start=start, end=end,
                           auto_adjust=True, progress=False)
        if raw.empty:
            import difflib
            # Known shorthand → yfinance symbol aliases
            ALIASES: dict[str, str] = {
                "LQ45":    "^JKLQ45",
                "IHSG":    "^JKSE",
                "JCI":     "^JKSE",
                "JKSE":    "^JKSE",
                "NIKKEI":  "^N225",
                "N225":    "^N225",
                "HSI":     "^HSI",
                "KOSPI":   "^KS11",
                "FTSE":    "^FTSE",
                "DAX":     "^GDAXI",
                "CAC40":   "^FCHI",
                "DJI":     "^DJI",
                "DJIA":    "^DJI",
                "SP500":   "^GSPC",
                "NASDAQ":  "^IXIC",
                "VIX":     "^VIX",
            }
            alias = ALIASES.get(ticker.upper())
            if alias:
                raise ValueError(
                    f"Ticker not found: '{ticker}'. Did you mean {alias}?"
                )
            common = [
                "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","BRK-B","JPM","V",
                "NVO","PFE","LLY","JNJ","ABBV","MRK","AMGN","GILD","BIIB","MRNA",
                "SPY","QQQ","IWM","DIA","GLD","TLT","BTC-USD","ETH-USD",
                "^JKLQ45","^JKSE","^GSPC","^IXIC","^DJI","^VIX",
            ]
            suggestion = difflib.get_close_matches(ticker.upper(), common, n=1, cutoff=0.6)
            hint = f" Did you mean {suggestion[0]}?" if suggestion else " Check the ticker symbol."
            raise ValueError(f"Ticker not found: '{ticker}'.{hint}")

        import pandas as pd
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        raw = raw.reset_index()
        raw.columns = [c.lower() for c in raw.columns]
        raw = raw.rename(columns={"date": "timestamp"})

        df = pl.from_pandas(raw).with_columns(
            pl.col("timestamp").cast(pl.Date).cast(pl.Datetime("us")),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
        )
        path = DATA_DIR / f"{ticker}_daily.parquet"
        df.write_parquet(path)
        return df

    @staticmethod
    def _fetch_minute_yf(ticker: str, start: dt.date, end: dt.date) -> pl.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("pip install yfinance") from exc

        raw = yf.download(ticker, start=start, end=end,
                           interval="1m", auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"No 1m data from yfinance: {ticker}")

        import pandas as pd
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        raw = raw.reset_index()
        raw.columns = [str(c).lower() for c in raw.columns]
        raw = raw.rename(columns={"datetime": "timestamp", "date": "timestamp"})

        return pl.from_pandas(raw).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us")),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
        )

    # ------------------------------------------------------------------ #
    # Synthetic 1-minute data generator (GBM + intraday patterns)          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _synthesise_minute(
        ticker: str,
        start:  dt.date,
        end:    dt.date,
        seed:   int = 0,
    ) -> pl.DataFrame:
        """
        Geometric Brownian Motion on 1-minute bars with realistic intraday
        volatility and volume patterns.

        Intraday volatility profile (U-shape)
        --------------------------------------
        Empirically, volatility is highest in the first and last 30 minutes
        of the trading day and lowest around noon.  We approximate this with
        a sinusoidal pattern:
            vol_scalar[m] = 1.0 + 0.8 × cos(π × m / 195)²
        where m ∈ [0, 389] is the minute-of-day index.
        Result: vol is ~1.8× baseline at open/close, ~1.0× at midday.

        Intraday volume profile (J + reverse-J → M-shape)
        --------------------------------------------------
        Volume is highest at open (large institutional orders, overnight
        news), dips at midday, then picks up again into close.
            vol_shape[m] = 0.5 + exp(−m/30) + exp(−(389−m)/30)
        Normalised so the daily sum equals the expected ADV.

        Pharma drift adjustment
        -----------------------
        NVO: +60% annual drift (GLP-1 rally proxy)
        PFE: −8% annual drift (post-COVID headwind proxy)
        These are intentionally directional to test momentum vs. reversion.
        """
        DRIFT_ANNUAL = {"NVO": 0.60, "PFE": -0.08}
        VOL_ANNUAL   = {"NVO": 0.28, "PFE": 0.22}
        # Use instance metadata so any ticker's price/volume is picked up
        # _synthesise_minute is a @staticmethod but we need meta here —
        # we resolve it via TICKER_META fallback or a module-level lookup.
        from .metadata import STATIC_META, _load_cache
        _all_meta = {**_load_cache(), **STATIC_META}
        _ticker_meta = _all_meta.get(ticker, {})
        BASE_PRICE   = float(_ticker_meta.get("approx_price_2024", _ticker_meta.get("approx_price", 50.0)))
        BASE_ADV     = int(_ticker_meta.get("avg_daily_volume", 1_000_000))

        rng = random.Random(seed + hash(ticker) % 1000)

        drift_annual = DRIFT_ANNUAL.get(ticker, 0.0)
        vol_annual   = VOL_ANNUAL.get(ticker, 0.20)
        dt_min       = 1 / (TRADING_DAYS * MINUTES_PER_DAY)   # fraction of year per bar
        drift_min    = drift_annual * dt_min
        vol_min      = vol_annual   * math.sqrt(dt_min)

        # Precompute intraday vol and volume scalars
        vol_scalars = [
            1.0 + 0.8 * math.cos(math.pi * m / 195) ** 2
            for m in range(MINUTES_PER_DAY)
        ]
        raw_vol_shape = [
            0.5 + math.exp(-m / 30) + math.exp(-(389 - m) / 30)
            for m in range(MINUTES_PER_DAY)
        ]
        vol_shape_sum = sum(raw_vol_shape)
        vol_shapes = [v / vol_shape_sum * BASE_ADV for v in raw_vol_shape]

        timestamps, opens_, highs, lows, closes, volumes = [], [], [], [], [], []

        price = BASE_PRICE
        day   = start
        one_min = dt.timedelta(minutes=1)

        while day < end:
            if day.weekday() >= 5:   # skip weekends
                day += dt.timedelta(days=1)
                continue

            session_open = dt.datetime(day.year, day.month, day.day, 9, 30)

            for m in range(MINUTES_PER_DAY):
                ts  = session_open + m * one_min
                vs  = vol_scalars[m]
                ret = drift_min + vol_min * vs * rng.gauss(0, 1)

                o  = price
                c  = price * math.exp(ret)
                # Simulate H/L as a fraction of bar volatility
                bar_range = abs(c - o) * (1.0 + rng.random() * 0.5)
                h  = max(o, c) + bar_range * 0.3
                lo = min(o, c) - bar_range * 0.3

                bar_vol = max(1, int(vol_shapes[m] * (0.7 + 0.6 * rng.random())))

                timestamps.append(ts)
                opens_.append(o)
                highs.append(h)
                lows.append(lo)
                closes.append(c)
                volumes.append(bar_vol)

                price = c

            day += dt.timedelta(days=1)

        return pl.DataFrame({
            "timestamp": pl.Series(timestamps, dtype=pl.Datetime("us")),
            "open":      pl.Series(opens_,    dtype=pl.Float64),
            "high":      pl.Series(highs,     dtype=pl.Float64),
            "low":       pl.Series(lows,      dtype=pl.Float64),
            "close":     pl.Series(closes,    dtype=pl.Float64),
            "volume":    pl.Series(volumes,   dtype=pl.Int64),
        })

    # ------------------------------------------------------------------ #
    # Schema normalisation                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise_schema(df: pl.DataFrame) -> pl.DataFrame:
        """Coerce parquet files with varying column names/types to canonical schema."""
        ts_col = next(
            (c for c in ("timestamp", "datetime", "date", "time") if c in df.columns),
            None,
        )
        if ts_col is None:
            raise ValueError("Parquet file must have a timestamp/date column.")
        if ts_col != "timestamp":
            df = df.rename({ts_col: "timestamp"})

        df = df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("us")),
        )
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64))
        if "volume" in df.columns:
            df = df.with_columns(pl.col("volume").cast(pl.Int64))
        return df

    # ------------------------------------------------------------------ #
    # Feature engineering                                                   #
    # ------------------------------------------------------------------ #

    def _engineer_features(
        self,
        df:     pl.DataFrame,
        ticker: str,
        start:  dt.date,
        end:    dt.date,
    ) -> pl.DataFrame:
        """
        Add all derived columns needed by the engine and cost model.

        Columns added (all PiT-safe via shift(1)):
            log_return       — ln(close_t / close_{t-1})
            true_range       — max(H−L, |H−prev_C|, |L−prev_C|)
            atr_14           — 14-bar EWM ATR (Wilder smoothing)
            avg_volume_20    — 20-bar rolling mean of volume
            realised_vol_20  — 20-bar rolling std of log_return (annualised)
            mean_ret_5/20/60 — rolling mean log_return (for signals)
            std_ret_20       — rolling std (for z-score)
            zscore_20d       — (lagged_ret − mean_ret_20) / std_ret_20
            momentum_60_5    — 60-bar cumulative return, skip last 5 bars
            is_event_window  — bool: within pre/post window of a binary event
        """
        # ── Log return ─────────────────────────────────────────────────── #
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1)).log()
            .alias("log_return")
        )

        # ── True Range & ATR-14 ────────────────────────────────────────── #
        prev_c = pl.col("close").shift(1)
        tr = pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - prev_c).abs(),
            (pl.col("low")  - prev_c).abs(),
        )
        df = df.with_columns(tr.alias("true_range")).with_columns(
            pl.col("true_range").ewm_mean(span=14, adjust=False).alias("atr_14")
        ).drop("true_range")

        # ── Volume rolling average (for cost model) ────────────────────── #
        vol_col = "volume" if "volume" in df.columns else None
        if vol_col:
            df = df.with_columns(
                pl.col("volume").shift(1).rolling_mean(window_size=20)
                .alias("avg_volume_20")
            )

        # ── Rolling return stats (PiT: shift(1) before rolling) ───────── #
        lagged_ret = pl.col("log_return").shift(1)
        ann = float(self.ann_factor)

        df = df.with_columns(
            (lagged_ret.rolling_std(window_size=20) * ann ** 0.5)
            .alias("realised_vol_20"),
        )
        for w in (5, 20, 60):
            df = df.with_columns(
                lagged_ret.rolling_mean(window_size=w).alias(f"mean_ret_{w}"),
                lagged_ret.rolling_std(window_size=w).alias(f"std_ret_{w}"),
            )

        # ── Z-score (mean-reversion signal input) ──────────────────────── #
        df = df.with_columns(
            (
                (lagged_ret - pl.col("mean_ret_20"))
                / pl.col("std_ret_20").clip(lower_bound=1e-9)
            ).alias("zscore_20d")
        )

        # ── Momentum (skip-5 window) ───────────────────────────────────── #
        df = df.with_columns(
            lagged_ret.shift(5).rolling_sum(window_size=55).alias("momentum_60_5")
        )

        # ── Binary event mask ──────────────────────────────────────────── #
        date_series = pl.col("timestamp").cast(pl.Date)
        df = df.with_columns(
            build_event_mask(
                pl.col("timestamp").cast(pl.Date),
                ticker, start, end,
            ).alias("is_event_window")
        )

        return df
