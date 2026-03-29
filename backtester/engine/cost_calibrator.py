"""
engine/cost_calibrator.py — Derive COST_PARAMS from a ticker's own price/volume history.

Instead of hardcoding base_bps and impact_coefficient per ticker, calibrate
them from the ticker's 30-bar median ATR and median daily volume:

  base_bps = clip(median_atr_pct × 100 × 0.15,  2.0, 20.0)
             ↑ spread ≈ 15% of daily range, capped at 2–20 bps

  impact_coefficient = clip(300_000 / √(median_volume), 5.0, 80.0)
             ↑ tighter markets (high vol) have lower κ; illiquid have higher

  event_spread_multiplier = 4.0   (universal — FDA/earnings events)
  base_liquidity = median_volume   (use actual observed liquidity)

Quant Why: different stocks have wildly different liquidity profiles.
SPY at 80M shares/day has κ ≈ 5 bps; a micro-cap at 200k shares/day
has κ ≈ 50 bps. A formula-based approach auto-calibrates to any ticker
without manual intervention.
"""

from __future__ import annotations

import polars as pl

from .costs import COST_PARAMS


# --------------------------------------------------------------------------- #
# Core calibration formula                                                      #
# --------------------------------------------------------------------------- #

def calibrate_from_df(ticker: str, df: pl.DataFrame) -> dict:
    """
    Derive COST_PARAMS-compatible dict from a ticker's price/volume DataFrame.

    Parameters
    ----------
    ticker : str
        Ticker symbol (used only for logging — not for lookup).
    df : pl.DataFrame
        Must contain columns: close, volume, atr_14.
        Typically the per-ticker slice from fetch_universe_long() after
        _ensure_columns() has been applied.

    Returns
    -------
    dict with keys: base_bps, impact_coefficient, event_spread_multiplier,
                    base_liquidity
    """
    # ── Median volume ─────────────────────────────────────────────────── #
    if "volume" in df.columns:
        raw_vol = df["volume"].cast(pl.Float64).median()
        median_volume = float(raw_vol) if raw_vol is not None and float(raw_vol) > 0 else 1_000_000.0
    else:
        median_volume = 1_000_000.0

    # ── Median ATR ────────────────────────────────────────────────────── #
    if "atr_14" in df.columns:
        raw_atr = df["atr_14"].median()
        if raw_atr is not None and float(raw_atr) > 0:
            median_atr = float(raw_atr)
        else:
            # Fallback: 2% of median close
            median_atr = float(df["close"].median()) * 0.02
    else:
        median_atr = float(df["close"].median()) * 0.02

    # ── Median price ──────────────────────────────────────────────────── #
    median_price = float(df["close"].median())
    if median_price <= 0:
        median_price = 1.0   # guard against zero/negative prices

    # ── Formula ───────────────────────────────────────────────────────── #
    median_atr_pct = median_atr / median_price

    # Spread ≈ 15% of daily range, capped at [2, 20] bps
    base_bps = max(2.0, min(20.0, median_atr_pct * 100.0 * 0.15))

    # Impact: illiquid stocks get higher κ, liquid ones lower
    impact_coefficient = max(5.0, min(80.0, 300_000.0 / (median_volume ** 0.5)))

    return {
        "base_bps": base_bps,
        "impact_coefficient": impact_coefficient,
        "event_spread_multiplier": 4.0,   # universal — FDA/earnings events
        "base_liquidity": median_volume,
    }


# --------------------------------------------------------------------------- #
# Public helpers                                                                #
# --------------------------------------------------------------------------- #

def get_cost_params(ticker: str, df: pl.DataFrame) -> dict:
    """
    Return cost params for a ticker: uses COST_PARAMS if the ticker is known,
    otherwise calibrates from the supplied DataFrame.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    df : pl.DataFrame
        Per-ticker price/volume DataFrame (used only when ticker is unknown).

    Returns
    -------
    dict compatible with compute_transaction_costs() keyword arguments.
    """
    if ticker in COST_PARAMS:
        return COST_PARAMS[ticker]
    return calibrate_from_df(ticker, df)


def calibrate_universe(
    tickers: list[str],
    frames: dict[str, pl.DataFrame],
) -> dict[str, dict]:
    """
    Calibrate cost params for every ticker in the universe.

    Parameters
    ----------
    tickers : list[str]
        All ticker symbols to process.
    frames : dict[str, pl.DataFrame]
        Mapping of ticker → per-ticker DataFrame.

    Returns
    -------
    dict[str, dict]
        Mapping of ticker → cost params dict.
    """
    return {t: get_cost_params(t, frames[t]) for t in tickers if t in frames}
