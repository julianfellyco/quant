"""backtester/strategy/pairs_scanner.py — Universe-wide cointegration scanner.

Scans all ticker pairs in a universe for statistical cointegration using:
  - Engle-Granger two-step test (statsmodels coint)
  - Mean-reversion half-life via AR(1) regression
  - Hurst exponent (R/S method) for mean-reversion confirmation
  - Spread volatility and correlation filters

Usage:
    scanner = PairsScanner(min_coint_pvalue=0.05, max_half_life=60)
    results = scanner.scan({"PFE": pfe_closes, "NVO": nvo_closes, ...})
    tradeable = [r for r in results if r.is_tradeable]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class PairResult:
    """Cointegration analysis result for a ticker pair."""

    ticker_a: str
    ticker_b: str
    coint_pvalue: float          # Engle-Granger p-value (lower = more cointegrated)
    half_life: float             # mean-reversion half-life in days
    hurst_exponent: float        # H < 0.5 = mean-reverting, > 0.5 = trending
    spread_volatility: float     # annualised spread vol
    correlation: float           # price correlation
    hedge_ratio: float           # OLS beta (B ≈ α + β·A)
    is_tradeable: bool           # passes all filters


class PairsScanner:
    """Scan a universe of tickers for cointegrated, mean-reverting pairs.

    Args:
        min_coint_pvalue: Reject pairs with p-value above this threshold.
        max_half_life:    Reject pairs whose spread reverts too slowly (days).
        min_half_life:    Reject pairs that revert too quickly (noise).
        max_hurst:        Reject pairs whose Hurst > this (not mean-reverting).
        lookback_days:    Number of bars to use for analysis.
    """

    def __init__(
        self,
        min_coint_pvalue: float = 0.05,
        max_half_life: float = 60.0,
        min_half_life: float = 5.0,
        max_hurst: float = 0.5,
        lookback_days: int = 252,
    ) -> None:
        self.min_coint_pvalue = min_coint_pvalue
        self.max_half_life = max_half_life
        self.min_half_life = min_half_life
        self.max_hurst = max_hurst
        self.lookback_days = lookback_days

    def scan(self, prices: dict[str, pl.Series]) -> list[PairResult]:
        """Scan all pairs in the universe for cointegration.

        Args:
            prices: dict mapping ticker → pl.Series of close prices (date-aligned)

        Returns:
            List of PairResult sorted by coint_pvalue ascending (best first).
        """
        from statsmodels.tsa.stattools import coint

        tickers = list(prices.keys())
        results: list[PairResult] = []

        for a, b in combinations(tickers, 2):
            pa = prices[a].drop_nulls().to_numpy()
            pb = prices[b].drop_nulls().to_numpy()

            min_len = min(len(pa), len(pb))
            if min_len < max(self.lookback_days // 2, 30):
                continue

            # Use last N days
            n = min(min_len, self.lookback_days)
            pa = pa[-n:]
            pb = pb[-n:]

            try:
                _, pvalue, _ = coint(pa, pb)
            except Exception as exc:
                logger.debug("coint failed for %s/%s: %s", a, b, exc)
                continue

            if pvalue > self.min_coint_pvalue:
                continue

            # OLS hedge ratio: pb = alpha + beta * pa + error
            beta, alpha = np.polyfit(pa, pb, 1)
            spread = pb - beta * pa

            # Half-life via AR(1): Δspread = ϕ·spread[t-1] + ε
            half_life = self._half_life(spread)

            # Hurst exponent
            hurst = self._hurst_rs(spread)

            # Annualised spread vol (std of daily changes)
            spread_vol = float(np.std(np.diff(spread)) * np.sqrt(252))

            # Price correlation
            corr = float(np.corrcoef(pa, pb)[0, 1])

            is_tradeable = (
                self.min_half_life <= half_life <= self.max_half_life
                and hurst < self.max_hurst
            )

            results.append(PairResult(
                ticker_a=a, ticker_b=b,
                coint_pvalue=float(pvalue),
                half_life=float(half_life),
                hurst_exponent=float(hurst),
                spread_volatility=spread_vol,
                correlation=corr,
                hedge_ratio=float(beta),
                is_tradeable=is_tradeable,
            ))

        results.sort(key=lambda r: r.coint_pvalue)
        return results

    @staticmethod
    def _half_life(spread: np.ndarray) -> float:
        """Compute mean-reversion half-life via AR(1) regression on spread."""
        if len(spread) < 4:
            return float("inf")
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)
        try:
            coef = np.polyfit(spread_lag, spread_diff, 1)
            phi = coef[0]
            if phi >= 0:
                return float("inf")
            return float(-np.log(2) / phi)
        except Exception:
            return float("inf")

    @staticmethod
    def _hurst_rs(series: np.ndarray, max_lag: int = 20) -> float:
        """Simplified R/S Hurst exponent.

        H < 0.5 → mean-reverting
        H ≈ 0.5 → random walk
        H > 0.5 → trending
        """
        lags = list(range(2, min(max_lag, len(series) // 4)))
        if len(lags) < 2:
            return 0.5
        rs_values = []
        for lag in lags:
            chunks = [series[i:i + lag] for i in range(0, len(series) - lag, lag)]
            rs = []
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                demeaned = chunk - np.mean(chunk)
                cumdev = np.cumsum(demeaned)
                r_range = np.max(cumdev) - np.min(cumdev)
                sd = np.std(chunk, ddof=1)
                if sd > 0:
                    rs.append(r_range / sd)
            if rs:
                rs_values.append(float(np.mean(rs)))
        if len(rs_values) < 2:
            return 0.5
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)
        try:
            hurst = float(np.polyfit(log_lags, log_rs, 1)[0])
        except Exception:
            hurst = 0.5
        return hurst
