"""
strategy/signals.py — Signal generators: Mean Reversion and Momentum.

Both functions return a pl.Series of {−1, 0, +1} values.
DataHandler.align_signals() then shifts by 1 bar to enforce PiT execution.

Hysteresis band (mean reversion)
---------------------------------
Entry at |z| > entry_z, exit at |z| < exit_z.  Without the band the strategy
churns on every tick of the z-score through the threshold, generating massive
turnover.  The band creates a sticky regime: once in a position you need the
z-score to collapse significantly before you exit.

Event hedge
-----------
When use_event_hedge=True, both strategies flatten to 0 on event-window bars.
Quant Why: a systematic strategy has no edge on binary-outcome events.
The cost model already penalises trading near events via the spread multiplier,
but the hedge prevents carrying an adverse directional position into a
coin-flip FDA decision.
"""

from __future__ import annotations

import polars as pl


def mean_reversion_signal(
    df:              pl.DataFrame,
    entry_z:         float = 1.5,
    exit_z:          float = 0.3,
    use_event_hedge: bool  = True,
) -> pl.Series:
    """
    Z-score mean-reversion signal with hysteresis band and event hedge.

    State machine (one-pass loop — avoids look-ahead from Polars .shift
    chains on state-dependent logic):
        z < −entry_z  → enter long  (+1)
        z >  entry_z  → enter short (−1)
        |z| < exit_z  → exit        (0)
        event window  → force flat  (0)
    """
    if "zscore_20d" not in df.columns:
        raise ValueError("Column 'zscore_20d' missing. Run DataHandler.load() first.")

    z_vals     = df["zscore_20d"].to_list()
    event_mask = (
        df["is_event_window"].to_list()
        if "is_event_window" in df.columns
        else [False] * len(df)
    )

    signals: list[float] = []
    state = 0.0

    for z, is_ev in zip(z_vals, event_mask):
        if is_ev and use_event_hedge:
            state = 0.0
        elif z is None:
            state = 0.0
        elif state == 0.0:
            if z < -entry_z:
                state = 1.0
            elif z > entry_z:
                state = -1.0
        else:
            if abs(z) < exit_z:
                state = 0.0
        signals.append(state)

    return pl.Series("signal", signals, dtype=pl.Float64)


def momentum_signal(
    df:              pl.DataFrame,
    use_event_hedge: bool = True,
) -> pl.Series:
    """
    Momentum signal: sign of 60-bar cumulative return (skip last 5 bars).

    The 5-bar skip avoids the well-documented short-term reversal that
    contaminates raw momentum: price that moved up in the last 5 bars often
    mean-reverts in the next 5, conflating two opposing effects.

    Quant Why for GLP-1 context: NVO's 2024 rally was a 12–18 month
    institutional re-rating.  The 60-bar lookback captures this slow-burn
    momentum while filtering out short-term noise.  A 252-bar lookback
    would also work but is slower to turn when the trend eventually reverses.
    """
    if "momentum_60_5" not in df.columns:
        raise ValueError("Column 'momentum_60_5' missing. Run DataHandler.load() first.")

    mom_vals   = df["momentum_60_5"].to_list()
    event_mask = (
        df["is_event_window"].to_list()
        if "is_event_window" in df.columns
        else [False] * len(df)
    )

    signals: list[float] = []
    for m, is_ev in zip(mom_vals, event_mask):
        if is_ev and use_event_hedge:
            signals.append(0.0)
        elif m is None:
            signals.append(0.0)
        elif m > 0:
            signals.append(1.0)
        else:
            signals.append(-1.0)

    return pl.Series("signal", signals, dtype=pl.Float64)
