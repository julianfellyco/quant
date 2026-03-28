"""api/routes/pairs.py — POST /api/pairs."""
from __future__ import annotations

import datetime as dt
import math

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
