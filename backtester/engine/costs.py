"""
engine/costs.py — Transaction cost model with event-driven spread spiking,
                  volume-based liquidity adjustment, and square-root market impact.

Cost decomposition
------------------
Total cost = spread_cost + market_impact

  spread_cost   = notional × (base_bps × event_multiplier) / 10_000
  market_impact = notional × α × √(|Q| / adjusted_lc) / 10_000

  adjusted_lc   = base_lc × volume_ratio          ← volume liquidity adjustment
  volume_ratio  = bar_volume / avg_bar_volume      ← clipped [0.1, 5.0]

Event spread spiking
--------------------
During binary events (FDA decisions, earnings, Phase III readouts), bid-ask
spreads blow out dramatically.  An FDA decision day for NVO can see spreads
widen from 3 bps to 30–60 bps in the minutes around the announcement.

We model this by multiplying base_bps by `event_spread_multiplier` on event
window bars.  This has two effects:
  1. Entering / exiting on event days is expensive → discourages gambling on
     binary outcomes with a naive systematic strategy.
  2. Holding through the event costs nothing in spread (position unchanged,
     no trade = no spread paid).  The cost is captured via price impact
     (adverse selection) in the log return itself.

Volume liquidity adjustment
---------------------------
Quant Why: Market impact is inversely proportional to liquidity.  For a
1-minute bar, liquidity is proxied by bar volume relative to its average.
A 9:31 bar with 5× normal volume (market-on-open rush) is far more liquid
than a 12:00 bar with 0.3× normal volume.

adjusted_lc = base_lc × (bar_volume / avg_volume).clip(0.1, 5.0)

If volume_ratio = 2.0  → adjusted_lc doubles → impact halves.
If volume_ratio = 0.2  → adjusted_lc shrinks → impact 5× higher.

For daily bars, bar_volume IS the daily volume, so avg_volume is the rolling
20-day average daily volume.
"""

from __future__ import annotations

import polars as pl


# --------------------------------------------------------------------------- #
# Per-ticker calibration                                                        #
# --------------------------------------------------------------------------- #

COST_PARAMS: dict[str, dict] = {
    "PFE": {
        # ≈1 cent spread on $28 stock ≈ 3.6 bps.  Add 1.4 bps fees → 5 bps.
        "base_bps":             5.0,
        # Square-root impact coefficient.  Calibrated so that at 10% ADV
        # (2.8M shares), impact = 16 × √0.10 ≈ 5 bps — matching empirical
        # estimates for large-cap US equities (Almgren-Chriss η ≈ 0.1).
        "impact_coefficient":  16.0,
        # 2024 average daily volume (shares). Source: Bloomberg consensus.
        "base_liquidity":   28_000_000,
        # Spread multiplier during event windows (earnings, FDA decisions).
        # PFE has less binary risk than NVO → lower multiplier.
        "event_spread_mult":    3.0,
    },
    "NVO": {
        # NVO ADR: wider spread (~5 bps) + ADR conversion fee (~2 bps) = 8 bps.
        "base_bps":             8.0,
        # Square-root impact coefficient.  At 10% ADV (350K shares),
        # impact = 25 × √0.10 ≈ 7.9 bps — consistent with ADR illiquidity.
        "impact_coefficient":  25.0,
        "base_liquidity":    3_500_000,
        # NVO has extreme binary event risk (GLP-1 trial readouts, FDA CVD
        # indication).  Spread during announcement bars can be 5–8× normal.
        "event_spread_mult":    6.0,
    },
}


# --------------------------------------------------------------------------- #
# Core cost function                                                            #
# --------------------------------------------------------------------------- #

def compute_transaction_costs(
    trade_sizes:         pl.Series,   # signed shares (+buy / −sell)
    prices:              pl.Series,   # bar close price
    atr:                 pl.Series,   # ATR-14 for regime scaling
    liquidity_constant:  float,       # base ADV proxy (shares/day or shares/bar)
    base_bps:            float,       # fixed spread + fee cost in bps
    impact_coefficient:  float,       # linear impact coefficient (bps at 1× LC)
    is_event:            pl.Series | None = None,   # boolean: True on event bars
    event_spread_mult:   float = 1.0,               # multiplier on base_bps
    bar_volume:          pl.Series | None = None,   # volume per bar
    avg_bar_volume:      pl.Series | None = None,   # rolling avg volume per bar
) -> pl.Series:
    """
    Compute per-bar transaction costs in USD.

    Two-stage calculation
    ---------------------
    Stage 1 — Spread + fees (per trade, pays only when position changes):
        effective_bps = base_bps × (event_spread_mult if is_event else 1.0)
        spread_cost   = |notional| × effective_bps / 10_000

    Stage 2 — Square-root market impact (per trade, volume-adjusted):
        volume_ratio  = (bar_volume / avg_bar_volume).clip(0.1, 5.0)
        adjusted_lc   = liquidity_constant × volume_ratio
        participation = |trade_size| / adjusted_lc
        impact_bps    = impact_coefficient × √participation
        atr_scalar    = (atr / median_atr).clip(0.5, 3.0)
        impact_cost   = |notional| × impact_bps × atr_scalar / 10_000

    Square-root law rationale (Almgren-Chriss)
    ------------------------------------------
    Empirically, market impact ∝ σ × √(Q/ADV).  Doubling trade size does NOT
    double impact — the exponent 0.5 is remarkably universal across asset classes.
    The linear model (exponent 1.0) is overly punitive on large trades and too
    lenient on small ones.  Using the square-root law means:
      - A 100× increase in trade size → 10× increase in impact bps (not 100×)
      - At very large sizes (Q ≈ ADV) impact is still bounded sub-linearly
    impact_coefficient is calibrated so that at 10% ADV the impact equals the
    empirically-observed values (PFE ≈ 5 bps, NVO ≈ 8 bps at 10% participation).

    Returns:
        Series[Float64] of non-negative USD costs, zero when trade_size == 0.
    """
    abs_trade = trade_sizes.abs()
    notional  = abs_trade * prices
    is_trade  = (abs_trade > 0).cast(pl.Float64)

    # ── Stage 1: Spread cost ──────────────────────────────────────────── #
    if is_event is not None:
        eff_bps = is_event.cast(pl.Float64) * (event_spread_mult - 1.0) * base_bps + base_bps
    else:
        eff_bps = pl.Series([base_bps] * len(trade_sizes))

    spread_cost = notional * eff_bps / 10_000

    # ── Stage 2: Volume-adjusted linear impact ────────────────────────── #
    if bar_volume is not None and avg_bar_volume is not None:
        safe_avg = avg_bar_volume.clip(lower_bound=1.0)
        vol_ratio = (bar_volume / safe_avg).clip(lower_bound=0.1, upper_bound=5.0)
    else:
        vol_ratio = pl.Series([1.0] * len(trade_sizes))

    adjusted_lc = pl.Series([float(liquidity_constant)] * len(trade_sizes)) * vol_ratio

    participation     = abs_trade / adjusted_lc
    impact_bps_series = impact_coefficient * participation.sqrt()

    # ATR regime: costs scale with realised volatility (proxy for spread widening)
    median_atr = atr.median() or 1e-9
    atr_scalar = (atr / median_atr).clip(lower_bound=0.5, upper_bound=3.0)

    impact_cost = notional * impact_bps_series / 10_000 * atr_scalar

    total = (spread_cost + impact_cost) * is_trade
    return total.alias("transaction_cost_usd")


def cost_for_ticker(
    ticker:      str,
    trade_sizes: pl.Series,
    prices:      pl.Series,
    atr:         pl.Series,
    is_event:    pl.Series | None = None,
    bar_volume:  pl.Series | None = None,
    avg_volume:  pl.Series | None = None,
) -> pl.Series:
    """Convenience wrapper: looks up calibration params by ticker."""
    p = COST_PARAMS.get(ticker)
    if p is None:
        raise ValueError(f"No cost params for '{ticker}'. Add to COST_PARAMS.")
    return compute_transaction_costs(
        trade_sizes        = trade_sizes,
        prices             = prices,
        atr                = atr,
        liquidity_constant = p["base_liquidity"],
        base_bps           = p["base_bps"],
        impact_coefficient = p["impact_coefficient"],
        is_event           = is_event,
        event_spread_mult  = p["event_spread_mult"],
        bar_volume         = bar_volume,
        avg_bar_volume     = avg_volume,
    )
