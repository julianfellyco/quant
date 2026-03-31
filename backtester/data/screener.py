"""
data/screener.py — Dynamic universe screener: builds ticker lists from
                   public data sources (Wikipedia, yfinance) or hardcoded sets.

Class overview
--------------
Screener.sp500()
    Scrapes the Wikipedia S&P 500 table and returns a sorted list of tickers.
    BRK.B → BRK-B style dot-to-dash replacement for yfinance compatibility.
    Result is cached in-process (_sp500_cache class variable).

Screener.nasdaq100()
    Same approach for Nasdaq-100 from the Nasdaq-100 Wikipedia article.

Screener.top_by_market_cap(n, universe)
    Fetches the chosen universe (sp500 or nasdaq100), then uses yfinance to
    retrieve recent market cap data. Falls back to alphabetical order if
    yfinance market cap lookup fails.

Screener.by_volume_filter(tickers, min_avg_volume, lookback_days)
    Given an explicit ticker list, filters to those whose recent average daily
    volume exceeds min_avg_volume. Uses yfinance.download() with threads=True.

Screener.pharma_large_cap()
    Returns a hardcoded list of ~25 major pharma/biotech tickers.
    No network calls required — safe to use offline.
"""

from __future__ import annotations

import logging
import warnings
from typing import ClassVar, Optional

logger = logging.getLogger(__name__)

# Hardcoded pharma/biotech universe — major global names, US-listed.
_PHARMA_LARGE_CAP: list[str] = [
    "PFE", "NVO", "LLY", "JNJ", "ABBV", "MRK", "BMY", "AMGN",
    "GILD", "BIIB", "REGN", "VRTX", "MRNA", "BNTX", "AZN",
    "GSK", "SNY", "NVS", "RHHBY", "TAK", "ALNY", "INCY",
    "SGEN", "HZNP", "RARE",
]


class Screener:
    """
    Static/class-method universe builder.

    All methods return list[str] of ticker symbols ready for use with
    fetch_universe() or yfinance.download().

    Wikipedia scraping uses pandas.read_html() which requires lxml:
        pip install pandas lxml

    Caching
    -------
    sp500() and nasdaq100() store results in class-level variables so
    repeated calls within the same process skip the network round-trip.
    """

    _sp500_cache:    ClassVar[Optional[list[str]]] = None
    _nasdaq100_cache: ClassVar[Optional[list[str]]] = None

    # ------------------------------------------------------------------ #
    # S&P 500                                                               #
    # ------------------------------------------------------------------ #

    @classmethod
    def sp500(cls) -> list[str]:
        """
        Fetch S&P 500 constituents from Wikipedia.

        Source: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
        The first table on that page has a 'Symbol' column.
        Dots are replaced with dashes (BRK.B → BRK-B) for yfinance.

        Returns
        -------
        list[str]
            Sorted list of ticker symbols.
        """
        if cls._sp500_cache is not None:
            return cls._sp500_cache

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for Screener.sp500(). "
                "Install with: pip install pandas lxml"
            )

        try:
            import io
            import urllib.request
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; quant-backtester/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8")
            tables = pd.read_html(io.StringIO(html))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch S&P 500 list from Wikipedia: {exc}."
            ) from exc

        # The first table contains constituents; 'Symbol' column has tickers.
        table = tables[0]
        symbol_col = next(
            (c for c in table.columns if "symbol" in str(c).lower()),
            None,
        )
        if symbol_col is None:
            raise ValueError(
                f"Could not find Symbol column in Wikipedia table. "
                f"Available columns: {list(table.columns)}"
            )

        tickers = (
            table[symbol_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)
            .tolist()
        )
        tickers = sorted(set(t for t in tickers if t and t != "nan"))
        cls._sp500_cache = tickers
        return tickers

    # ------------------------------------------------------------------ #
    # Nasdaq-100                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def nasdaq100(cls) -> list[str]:
        """
        Fetch Nasdaq-100 constituents from Wikipedia.

        Source: https://en.wikipedia.org/wiki/Nasdaq-100
        Looks for a table containing a 'Ticker' or 'Symbol' column.

        Returns
        -------
        list[str]
            Sorted list of ticker symbols.
        """
        if cls._nasdaq100_cache is not None:
            return cls._nasdaq100_cache

        url = "https://en.wikipedia.org/wiki/Nasdaq-100"

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for Screener.nasdaq100(). "
                "Install with: pip install pandas lxml"
            )

        try:
            import io
            import urllib.request
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; quant-backtester/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8")
            tables = pd.read_html(io.StringIO(html))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch Nasdaq-100 list from Wikipedia: {exc}."
            ) from exc

        # Find the table with a ticker/symbol column
        tickers: list[str] = []
        for table in tables:
            col_lower = {str(c).lower(): c for c in table.columns}
            symbol_col = (
                col_lower.get("ticker")
                or col_lower.get("symbol")
                or col_lower.get("ticker symbol")
            )
            if symbol_col is not None and len(table) > 50:
                tickers = (
                    table[symbol_col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.replace(".", "-", regex=False)
                    .tolist()
                )
                tickers = sorted(set(t for t in tickers if t and t != "nan"))
                break

        if not tickers:
            raise ValueError(
                "Could not find a Nasdaq-100 ticker table on Wikipedia. "
                "The page structure may have changed."
            )

        cls._nasdaq100_cache = tickers
        return tickers

    # ------------------------------------------------------------------ #
    # Top N by market cap                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def top_by_market_cap(cls, n: int = 100, universe: str = "sp500") -> list[str]:
        """
        Return the top n tickers by approximate market cap from the given universe.

        Fetches the universe ticker list, then uses yfinance.Ticker.fast_info
        to retrieve market_cap for each ticker. Falls back to alphabetical
        order if yfinance calls fail.

        Parameters
        ----------
        n : int
            Number of tickers to return. Default 100.
        universe : str
            "sp500" or "nasdaq100". Default "sp500".

        Returns
        -------
        list[str]
            Top n tickers sorted by market cap descending.
        """
        if universe == "sp500":
            all_tickers = cls.sp500()
        elif universe == "nasdaq100":
            all_tickers = cls.nasdaq100()
        else:
            raise ValueError(f"universe must be 'sp500' or 'nasdaq100', got {universe!r}")

        try:
            import yfinance as yf
        except ImportError:
            warnings.warn(
                "yfinance not installed. Falling back to alphabetical order. "
                "pip install yfinance"
            )
            return sorted(all_tickers)[:n]

        # Attempt to fetch market caps via yfinance
        market_caps: dict[str, float] = {}
        try:
            # Use yfinance.download for a fast recent-close proxy, then
            # multiply by shares outstanding from fast_info.
            # For very large universes (500+), batch-download closes then
            # iterate fast_info for shares outstanding.
            import yfinance as yf

            # Batch download recent closes (1 week) to get prices fast
            import warnings as _warn
            with _warn.catch_warnings():
                _warn.simplefilter("ignore")
                price_data = yf.download(
                    all_tickers,
                    period="5d",
                    interval="1d",
                    threads=True,
                    progress=False,
                    auto_adjust=True,
                )

            # Extract latest close per ticker
            closes: dict[str, float] = {}
            if hasattr(price_data.columns, "levels"):
                # MultiIndex: (Price, Ticker)
                if "Close" in price_data.columns.get_level_values(0):
                    close_df = price_data["Close"]
                    for ticker in all_tickers:
                        if ticker in close_df.columns:
                            series = close_df[ticker].dropna()
                            if not series.empty:
                                closes[ticker] = float(series.iloc[-1])
            else:
                if "Close" in price_data.columns or "close" in price_data.columns:
                    col = "Close" if "Close" in price_data.columns else "close"
                    closes[all_tickers[0]] = float(price_data[col].dropna().iloc[-1])

            # Get shares outstanding from fast_info (one ticker at a time)
            for ticker in all_tickers:
                try:
                    t_obj = yf.Ticker(ticker)
                    shares = getattr(t_obj.fast_info, "shares", None)
                    price  = closes.get(ticker)
                    if shares and price:
                        market_caps[ticker] = shares * price
                except Exception:
                    pass

        except Exception as exc:
            logger.warning("top_by_market_cap: yfinance market cap fetch failed: %s", exc)

        if not market_caps:
            warnings.warn(
                "Could not retrieve market cap data from yfinance. "
                "Falling back to alphabetical order."
            )
            return sorted(all_tickers)[:n]

        # Sort descending by market cap; for tickers without data, rank last
        sorted_tickers = sorted(
            all_tickers,
            key=lambda t: market_caps.get(t, -1),
            reverse=True,
        )
        return sorted_tickers[:n]

    # ------------------------------------------------------------------ #
    # Volume filter                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def by_volume_filter(
        cls,
        tickers:        list[str],
        min_avg_volume: int = 1_000_000,
        lookback_days:  int = 20,
    ) -> list[str]:
        """
        Filter tickers to those with sufficient average daily trading volume.

        Downloads recent daily OHLCV using yfinance.download() with
        threads=True, computes the trailing average volume over lookback_days,
        and returns tickers where avg_volume >= min_avg_volume.

        Parameters
        ----------
        tickers : list[str]
            Candidate ticker list.
        min_avg_volume : int
            Minimum average daily volume. Default 1,000,000 shares/day.
        lookback_days : int
            Rolling window for average volume computation. Default 20.

        Returns
        -------
        list[str]
            Filtered list of tickers passing the volume threshold.
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for Screener.by_volume_filter(). "
                "pip install yfinance"
            )

        import warnings as _warn
        with _warn.catch_warnings():
            _warn.simplefilter("ignore")
            data = yf.download(
                tickers,
                period=f"{lookback_days + 5}d",   # extra buffer for non-trading days
                interval="1d",
                threads=True,
                progress=False,
                auto_adjust=True,
            )

        if data.empty:
            warnings.warn("by_volume_filter: yfinance returned empty data.")
            return []

        passing: list[str] = []

        # Handle both single-ticker (flat columns) and multi-ticker (MultiIndex)
        has_multiindex = hasattr(data.columns, "levels")

        for ticker in tickers:
            try:
                if has_multiindex:
                    if "Volume" in data.columns.get_level_values(0):
                        vol_series = data["Volume"][ticker].dropna()
                    else:
                        continue
                else:
                    # Single ticker download
                    col = "Volume" if "Volume" in data.columns else "volume"
                    if col not in data.columns:
                        continue
                    vol_series = data[col].dropna()

                if len(vol_series) == 0:
                    continue

                avg_vol = float(vol_series.tail(lookback_days).mean())
                if avg_vol >= min_avg_volume:
                    passing.append(ticker)
            except Exception as exc:
                logger.debug("by_volume_filter: skipping %s: %s", ticker, exc)

        return passing

    # ------------------------------------------------------------------ #
    # Pharma / biotech hardcoded list                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def pharma_large_cap() -> list[str]:
        """
        Return a hardcoded list of ~25 major pharma/biotech tickers.

        Includes US-listed ADRs (NVO, AZN, GSK, SNY, NVS, RHHBY, TAK) and
        domestic large/mid caps. No network calls required.

        Returns
        -------
        list[str]
            List of ticker symbols (all uppercase, yfinance-compatible).
        """
        return list(_PHARMA_LARGE_CAP)
