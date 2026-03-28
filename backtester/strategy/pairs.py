"""
strategy/pairs.py — Pairs trading via rolling cointegration spread.

Lead-lag between NVO and PFE in the GLP-1 trade
-------------------------------------------------
In 2024–2025, Novo Nordisk (NVO) was the structural leader of the GLP-1
"weight-loss" theme.  Every positive trial readout, FDA approval, or supply
update moved NVO first; PFE's Danuglipron pipeline reacted secondarily.
This created two exploitable relative-value dynamics:

  1. Mean Reversion:  when the NVO/PFE spread widens beyond historical norms,
     it tends to revert.  The relative-value trade (long laggard / short leader)
     is market-neutral to the macro GLP-1 factor.

  2. Lead-lag:  NVO's log return on day T is weakly predictive of PFE's return
     on day T+1, reflecting the delayed information flow in the cross-ticker
     cointegration residual.

Cointegration methodology (rolling OLS)
-----------------------------------------
We estimate a rolling hedge ratio β:

    log_price_PFE(t) = α(t) + β(t) · log_price_NVO(t) + ε(t)

β is computed via rolling OLS (no scipy dependency):

    β = Cov(log_PFE, log_NVO) / Var(log_NVO)

using rolling window statistics.  The residual ε (the "spread") is then
Z-scored over a second rolling window.  The Z-scored spread is stationary
(by construction of the cointegration) and reverts to zero.

Signal rules
------------
    spread_z > +entry_z  →  NVO expensive vs PFE
                             short NVO (nvo_signal = −1), long PFE (pfe_signal = +1)
    spread_z < −entry_z  →  NVO cheap vs PFE
                             long NVO (nvo_signal = +1), short PFE (pfe_signal = −1)
    |spread_z| < exit_z  →  exit both legs (signals = 0)

The strategy is market-neutral in expectation: it profits from spread
reversion regardless of the overall GLP-1 sector direction.
"""

from __future__ import annotations

import math
from typing import Optional

import polars as pl


# --------------------------------------------------------------------------- #
# Spread computation                                                             #
# --------------------------------------------------------------------------- #

def compute_spread_zscore(
    nvo_df:          pl.DataFrame,
    pfe_df:          pl.DataFrame,
    hedge_window:    int = 60,
    zscore_window:   int = 20,
    price_col:       str = "close",
    timestamp_col:   str = "timestamp",
) -> pl.DataFrame:
    """
    Align NVO and PFE price series and compute the cointegration spread Z-score.

    Parameters
    ----------
    nvo_df, pfe_df : pl.DataFrame
        Must contain `timestamp_col` and `price_col` columns.
        Timestamps are inner-joined so the output only covers the overlap period.
    hedge_window : int
        Rolling window (bars) for the OLS hedge ratio β.
        Longer window → more stable β, slower to adapt to regime change.
        Default 60 bars ≈ 3 months of daily data.
    zscore_window : int
        Rolling window for the spread mean and std used in Z-scoring.
        Default 20 bars ≈ 1 month of daily data.
    price_col : str
        Column containing the price to use (close or mid-price).
    timestamp_col : str
        Datetime/Date column for alignment.

    Returns
    -------
    pl.DataFrame with columns:
        timestamp     — aligned timestamps
        nvo_close     — NVO price
        pfe_close     — PFE price
        log_nvo       — natural log of NVO price
        log_pfe       — natural log of PFE price
        beta          — rolling hedge ratio (β)
        alpha         — rolling intercept (α = mean_log_pfe − β × mean_log_nvo)
        spread        — residual: log_pfe − α − β × log_nvo
        spread_mean   — rolling mean of spread
        spread_std    — rolling std of spread
        spread_z      — Z-scored spread (stationarised)
    """
    # ── Align on timestamp ───────────────────────────────────────────── #
    nvo_sel = nvo_df.select([
        pl.col(timestamp_col),
        pl.col(price_col).alias("nvo_close"),
    ])
    pfe_sel = pfe_df.select([
        pl.col(timestamp_col),
        pl.col(price_col).alias("pfe_close"),
    ])

    df = nvo_sel.join(pfe_sel, on=timestamp_col, how="inner").sort(timestamp_col)

    # ── Log prices ───────────────────────────────────────────────────── #
    df = df.with_columns([
        pl.col("nvo_close").log(base=math.e).alias("log_nvo"),
        pl.col("pfe_close").log(base=math.e).alias("log_pfe"),
    ])

    # ── Rolling OLS hedge ratio ───────────────────────────────────────── #
    # β = Cov(log_pfe, log_nvo) / Var(log_nvo)
    # Cov(X, Y) = E[XY] − E[X]E[Y]  (via rolling means)
    # Var(Y)    = E[Y²] − E[Y]²
    df = df.with_columns([
        (pl.col("log_nvo") * pl.col("log_pfe"))
        .rolling_mean(window_size=hedge_window).alias("_roll_nvo_pfe"),
        pl.col("log_nvo").rolling_mean(window_size=hedge_window).alias("_roll_nvo"),
        pl.col("log_pfe").rolling_mean(window_size=hedge_window).alias("_roll_pfe"),
        (pl.col("log_nvo") * pl.col("log_nvo"))
        .rolling_mean(window_size=hedge_window).alias("_roll_nvo2"),
    ]).with_columns([
        (pl.col("_roll_nvo_pfe") - pl.col("_roll_nvo") * pl.col("_roll_pfe"))
        .alias("_cov"),
        (pl.col("_roll_nvo2") - pl.col("_roll_nvo") ** 2)
        .clip(lower_bound=1e-12).alias("_var_nvo"),
    ]).with_columns([
        (pl.col("_cov") / pl.col("_var_nvo")).alias("beta"),
    ]).with_columns([
        # α = mean(log_pfe) − β × mean(log_nvo)
        (pl.col("_roll_pfe") - pl.col("beta") * pl.col("_roll_nvo")).alias("alpha"),
    ])

    # ── Spread residual ──────────────────────────────────────────────── #
    df = df.with_columns([
        (pl.col("log_pfe") - pl.col("alpha") - pl.col("beta") * pl.col("log_nvo"))
        .alias("spread"),
    ])

    # ── Z-score the spread ───────────────────────────────────────────── #
    df = df.with_columns([
        pl.col("spread").rolling_mean(window_size=zscore_window).alias("spread_mean"),
        pl.col("spread").rolling_std(window_size=zscore_window).alias("spread_std"),
    ]).with_columns([
        ((pl.col("spread") - pl.col("spread_mean"))
         / pl.col("spread_std").clip(lower_bound=1e-9))
        .alias("spread_z"),
    ])

    # ── Drop intermediate columns ────────────────────────────────────── #
    return df.drop([
        "_roll_nvo_pfe", "_roll_nvo", "_roll_pfe", "_roll_nvo2", "_cov", "_var_nvo"
    ])


# --------------------------------------------------------------------------- #
# Signal generation                                                             #
# --------------------------------------------------------------------------- #

def pairs_signal(
    spread_df:       pl.DataFrame,
    entry_z:         float = 1.5,
    exit_z:          float = 0.3,
    use_event_hedge: bool  = True,
) -> pl.DataFrame:
    """
    Generate paired long/short signals based on the cointegration spread Z-score.

    Parameters
    ----------
    spread_df : pl.DataFrame
        Output of compute_spread_zscore().  Must contain 'spread_z'.
    entry_z : float
        Z-score threshold to enter a position.  Higher → fewer, higher-conviction
        trades.  Default 1.5 (≈ 93rd percentile under Gaussian).
    exit_z : float
        Z-score threshold to exit (hysteresis band).  Must be < entry_z.
    use_event_hedge : bool
        If True and 'is_event_window' column is present, both legs are forced
        to 0 on event-window bars.

    Returns
    -------
    pl.DataFrame with original columns plus:
        nvo_signal : Float64  {−1, 0, +1}
            +1 = long NVO, −1 = short NVO
        pfe_signal : Float64  {−1, 0, +1}
            +1 = long PFE, −1 = short PFE
        pair_state : str
            Human-readable regime ("long_spread" / "short_spread" / "flat")

    Signal logic
    ------------
    When spread_z > +entry_z  →  NVO too expensive vs PFE:
        nvo_signal = −1  (short the leader)
        pfe_signal = +1  (long the laggard)
        state = "short_spread"

    When spread_z < −entry_z  →  NVO too cheap vs PFE:
        nvo_signal = +1  (long the leader)
        pfe_signal = −1  (short the laggard)
        state = "long_spread"

    When |spread_z| < exit_z  →  spread has reverted:
        both signals = 0
        state = "flat"
    """
    if "spread_z" not in spread_df.columns:
        raise ValueError("Column 'spread_z' missing.  Run compute_spread_zscore() first.")

    z_vals     = spread_df["spread_z"].to_list()
    event_mask = (
        spread_df["is_event_window"].to_list()
        if "is_event_window" in spread_df.columns
        else [False] * len(spread_df)
    )

    nvo_sigs: list[float] = []
    pfe_sigs: list[float] = []
    states:   list[str]   = []

    nvo_state = 0.0
    pfe_state = 0.0

    for z, is_ev in zip(z_vals, event_mask):
        if is_ev and use_event_hedge:
            nvo_state = 0.0
            pfe_state = 0.0
        elif z is None or (isinstance(z, float) and math.isnan(z)):
            nvo_state = 0.0
            pfe_state = 0.0
        elif nvo_state == 0.0:
            # Entry logic
            if z > entry_z:
                # Spread too wide: NVO expensive → short NVO, long PFE
                nvo_state = -1.0
                pfe_state =  1.0
            elif z < -entry_z:
                # Spread too narrow: NVO cheap → long NVO, short PFE
                nvo_state =  1.0
                pfe_state = -1.0
        else:
            # Exit logic (hysteresis band)
            if abs(z) < exit_z:
                nvo_state = 0.0
                pfe_state = 0.0

        nvo_sigs.append(nvo_state)
        pfe_sigs.append(pfe_state)

        if nvo_state == 0.0:
            states.append("flat")
        elif nvo_state == -1.0:
            states.append("short_spread")
        else:
            states.append("long_spread")

    return spread_df.with_columns([
        pl.Series("nvo_signal", nvo_sigs, dtype=pl.Float64),
        pl.Series("pfe_signal", pfe_sigs, dtype=pl.Float64),
        pl.Series("pair_state", states,   dtype=pl.Utf8),
    ])


# --------------------------------------------------------------------------- #
# Diagnostics                                                                   #
# --------------------------------------------------------------------------- #

def spread_summary(spread_df: pl.DataFrame) -> dict:
    """
    Return a summary dict of spread statistics for diagnostic use.

    Keys: mean_beta, std_beta, spread_z_mean, spread_z_std,
          pct_long_spread, pct_short_spread, pct_flat
    """
    z = spread_df["spread_z"].drop_nulls()
    beta = spread_df["beta"].drop_nulls()

    if "pair_state" in spread_df.columns:
        n = len(spread_df)
        vc = spread_df["pair_state"].value_counts()
        counts = {row["pair_state"]: row["count"] for row in vc.iter_rows(named=True)}
        pct_long  = counts.get("long_spread",  0) / n
        pct_short = counts.get("short_spread", 0) / n
        pct_flat  = counts.get("flat",         0) / n
    else:
        pct_long = pct_short = pct_flat = float("nan")

    return {
        "mean_beta":         float(beta.mean() or 0.0),
        "std_beta":          float(beta.std()  or 0.0),
        "spread_z_mean":     float(z.mean() or 0.0),
        "spread_z_std":      float(z.std()  or 0.0),
        "pct_long_spread":   pct_long,
        "pct_short_spread":  pct_short,
        "pct_flat":          pct_flat,
    }
