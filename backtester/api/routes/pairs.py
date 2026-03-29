"""api/routes/pairs.py — POST /api/pairs."""
from __future__ import annotations

import datetime as dt
import math

import polars as pl
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backtester.data.handler import DataHandler
from backtester.strategy.pairs import (
    compute_spread_zscore, pairs_signal, spread_summary,
)

from api.deps import clean_float, granularity_enum, ts_to_str

router = APIRouter()


class PairsRequest(BaseModel):
    start_date:      str   = "2024-01-01"
    end_date:        str   = "2025-01-01"
    granularity:     str   = "daily"
    hedge_window:    int   = 60
    zscore_window:   int   = 20
    entry_z:         float = 1.5
    exit_z:          float = 0.3
    use_event_hedge: bool  = True


class PairsSummaryOut(BaseModel):
    mean_beta:        float | None
    std_beta:         float | None
    spread_z_mean:    float | None
    spread_z_std:     float | None
    pct_long_spread:  float | None
    pct_short_spread: float | None
    pct_flat:         float | None


class PairsResponse(BaseModel):
    spread_data: list[dict]
    summary:     PairsSummaryOut


@router.post("/pairs", response_model=PairsResponse)
def run_pairs(req: PairsRequest) -> PairsResponse:
    try:
        start = dt.date.fromisoformat(req.start_date)
        end   = dt.date.fromisoformat(req.end_date)
    except ValueError as e:
        raise HTTPException(400, f"Invalid date: {e}")

    try:
        gran = granularity_enum(req.granularity)
    except ValueError as e:
        raise HTTPException(400, str(e))

    handler = DataHandler(["NVO", "PFE"], granularity=gran)
    handler.load(start=start, end=end)

    nvo_df = handler["NVO"]
    pfe_df = handler["PFE"]

    spread_df = compute_spread_zscore(
        nvo_df,
        pfe_df,
        hedge_window  = req.hedge_window,
        zscore_window = req.zscore_window,
    )

    result_df = pairs_signal(
        spread_df,
        entry_z         = req.entry_z,
        exit_z          = req.exit_z,
        use_event_hedge = req.use_event_hedge,
    )

    summary = spread_summary(result_df)

    # Serialize spread_data rows
    rows = result_df.to_dicts()
    for row in rows:
        row["timestamp"] = ts_to_str(row["timestamp"])
        for k, v in row.items():
            if isinstance(v, float) and not math.isfinite(v):
                row[k] = None

    return PairsResponse(
        spread_data = rows,
        summary     = PairsSummaryOut(
            mean_beta        = clean_float(summary["mean_beta"]),
            std_beta         = clean_float(summary["std_beta"]),
            spread_z_mean    = clean_float(summary["spread_z_mean"]),
            spread_z_std     = clean_float(summary["spread_z_std"]),
            pct_long_spread  = clean_float(summary["pct_long_spread"]),
            pct_short_spread = clean_float(summary["pct_short_spread"]),
            pct_flat         = clean_float(summary["pct_flat"]),
        ),
    )


# ── Pairs Scanner ──────────────────────────────────────────────────────────── #

class PairsScanRequest(BaseModel):
    universe: str = "pharma"             # "pharma" | "tech" | "energy" | "custom"
    custom_tickers: list[str] = []
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    lookback_days: int = 252
    min_coint_pvalue: float = 0.05
    max_half_life: float = 60.0
    min_half_life: float = 5.0
    max_hurst: float = 0.5


class PairResultOut(BaseModel):
    ticker_a: str
    ticker_b: str
    coint_pvalue: float
    half_life: float
    hurst_exponent: float
    spread_volatility: float
    correlation: float
    hedge_ratio: float
    is_tradeable: bool


class ScanMetadata(BaseModel):
    universe_size: int
    pairs_tested: int
    pairs_cointegrated: int
    pairs_tradeable: int


class PairsScanResponse(BaseModel):
    pairs: list[PairResultOut]
    scan_metadata: ScanMetadata


@router.post("/pairs/scan", response_model=PairsScanResponse)
def scan_pairs(req: PairsScanRequest) -> PairsScanResponse:
    """Scan a ticker universe for cointegrated pairs."""
    from backtester.data.universe import UNIVERSES
    from backtester.data.universe_fetcher import fetch_universe_long
    from backtester.strategy.pairs_scanner import PairsScanner

    # Resolve tickers
    if req.universe == "custom":
        tickers = [t.strip().upper() for t in req.custom_tickers if t.strip()]
        if len(tickers) < 2:
            raise HTTPException(400, "custom_tickers must have at least 2 tickers")
    elif req.universe in UNIVERSES:
        tickers = UNIVERSES[req.universe].tickers
    else:
        raise HTTPException(400, f"Unknown universe '{req.universe}'. Use: pharma, tech, energy, custom")

    # Fetch prices
    try:
        long_df = fetch_universe_long(
            tickers,
            start=req.start_date,
            end=req.end_date,
            granularity="daily",
            max_workers=10,
        )
    except Exception as e:
        raise HTTPException(400, f"Data fetch failed: {e}")

    if long_df.is_empty():
        raise HTTPException(400, "No data returned for any ticker.")

    # Build price dict: ticker → close Series
    prices: dict = {}
    for ticker in tickers:
        subset = long_df.filter(pl.col("ticker") == ticker).sort("timestamp")
        if not subset.is_empty():
            prices[ticker] = subset["close"]

    if len(prices) < 2:
        raise HTTPException(400, "Need at least 2 tickers with data to scan for pairs.")

    # Run scanner
    scanner = PairsScanner(
        min_coint_pvalue=req.min_coint_pvalue,
        max_half_life=req.max_half_life,
        min_half_life=req.min_half_life,
        max_hurst=req.max_hurst,
        lookback_days=req.lookback_days,
    )
    from api.deps import clean_float
    results = scanner.scan(prices)

    n_tickers = len(prices)
    n_pairs = n_tickers * (n_tickers - 1) // 2

    return PairsScanResponse(
        pairs=[
            PairResultOut(
                ticker_a=r.ticker_a,
                ticker_b=r.ticker_b,
                coint_pvalue=r.coint_pvalue,
                half_life=r.half_life,
                hurst_exponent=r.hurst_exponent,
                spread_volatility=r.spread_volatility,
                correlation=r.correlation,
                hedge_ratio=r.hedge_ratio,
                is_tradeable=r.is_tradeable,
            )
            for r in results
        ],
        scan_metadata=ScanMetadata(
            universe_size=n_tickers,
            pairs_tested=n_pairs,
            pairs_cointegrated=len(results),
            pairs_tradeable=sum(1 for r in results if r.is_tradeable),
        ),
    )
