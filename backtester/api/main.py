"""api/main.py — FastAPI application entry point.

Start:
    cd /Users/julianfellyco/backtester
    PYTHONPATH=.. uvicorn api.main:app --reload --port 8000

Production (after `cd webapp && npm run build`):
    PYTHONPATH=.. uvicorn api.main:app --port 8000
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import backtest, pairs, stress, tickers, universe, walkforward
from backtester.api.routes.paper_trading import router as paper_router

STATIC_DIR = Path(__file__).parent.parent / "static"

app = FastAPI(
    title       = "Pharma Backtester API",
    description = "GLP-1 alpha engine: NVO/PFE backtest, pairs trading, stress test, walk-forward",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["http://localhost:5173"],   # Vite dev server
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

app.include_router(tickers.router,     prefix="/api", tags=["tickers"])
app.include_router(backtest.router,    prefix="/api", tags=["backtest"])
app.include_router(pairs.router,       prefix="/api", tags=["pairs"])
app.include_router(stress.router,      prefix="/api", tags=["stress"])
app.include_router(walkforward.router, prefix="/api", tags=["walkforward"])
app.include_router(universe.router,   prefix="/api", tags=["universe"])
app.include_router(paper_router,      prefix="/api", tags=["paper_trading"])

# ── Production: serve React app ─────────────────────────────────────────── #
if STATIC_DIR.exists():
    _assets = STATIC_DIR / "assets"
    if _assets.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
def spa_fallback(full_path: str) -> FileResponse:
    """Serve index.html for all non-API routes (React SPA routing)."""
    if full_path.startswith("api/"):
        from fastapi import HTTPException
        raise HTTPException(404)
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    from fastapi import HTTPException
    raise HTTPException(503, detail="Frontend not built. Run: cd webapp && npm run build")
