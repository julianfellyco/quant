# quant

Institutional-grade quantitative research toolkit covering a **Level-2 LOB Simulator** and a **Pharma Backtester** with a full-stack web UI.

---

## Repository Layout

```
quant/
├── backtester/          # Vectorised backtest engine + FastAPI + React webapp
│   ├── api/             # FastAPI routes (backtest, pairs, stress, walkforward, tickers)
│   ├── data/            # DataHandler (parquet + yfinance), events calendar, fetcher
│   ├── engine/          # VectorizedEngine, TransactionCost, SlippageModel, stress, walkforward
│   ├── stats/           # Metrics (Sharpe, Sortino, MDD, event decomp)
│   ├── strategy/        # momentum_signal, mean_reversion_signal, pairs cointegration
│   ├── tests/           # pytest suite (engine, execution, upgrade tests)
│   ├── webapp/          # React 18 + Vite 5 + TypeScript + Tailwind frontend
│   ├── static/          # Pre-built React output (served by FastAPI in production)
│   └── run_backtest.py  # CLI entry point
└── lob_simulator/       # Price-time priority LOB with OFI tracker
    ├── core/            # OrderBook, Level, Order, types
    ├── metrics/         # ExecutionMetrics, OFITracker
    └── tests/           # pytest suite
```

---

## Backtester

### Features

| Module | Description |
|---|---|
| **VectorizedEngine** | Polars-native backtest loop; momentum and mean-reversion signals |
| **Square-root impact** | Almgren-Chriss model: `impact_bps = κ × √(Q/ADV)` — calibrated 5 bps (PFE) / 8 bps (NVO) at 10% ADV |
| **Pairs trading** | Rolling OLS cointegration (NVO/PFE); market-neutral spread Z-score signals |
| **Monte Carlo stress** | Event-date shuffling ±N days, fragility score = P(Sharpe < 0) |
| **Walk-forward optimizer** | Rolling train/test grid search; IS/OOS Sharpe degradation + stability score |
| **Event decomposition** | Splits P&L into event-window vs non-event-window periods |

### Quickstart

```bash
cd backtester

# Install (use uv or pip)
pip install -e ".[dev]"
pip install fastapi uvicorn[standard] httpx

# Run CLI backtest
python run_backtest.py

# Start API server (serves React SPA at /)
PYTHONPATH=.. uvicorn api.main:app --port 8000 --reload

# Dev frontend (hot-reload, proxies /api → localhost:8000)
cd webapp && npm install && npm run dev

# Production build (output → backtester/static/)
cd webapp && npm run build
```

### Backtest Results (2024 full year, daily, $100k capital, 100 shares/unit)

| Ticker | Strategy | Net Sharpe | Total Return | MDD |
|---|---|---|---|---|
| NVO | Momentum | 0.77 | +30.2% | −26.4% |
| PFE | Momentum | 0.93 | +23.9% | −7.7% |
| NVO | Mean Reversion | −0.02 | +4.4% | — |
| PFE | Mean Reversion | −2.48 | −22.0% | — |

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/tickers` | Ticker metadata + binary event calendar |
| `POST` | `/api/backtest` | Run vectorised backtest (multi-ticker, multi-strategy) |
| `POST` | `/api/pairs` | NVO/PFE cointegration spread analysis |
| `POST` | `/api/stress` | Monte Carlo event-shuffling stress test |
| `POST` | `/api/walkforward` | Walk-forward parameter optimisation |
| `GET` | `/*` | React SPA fallback |

Interactive docs: **http://localhost:8000/docs**

---

## LOB Simulator

Price-time priority limit order book with real-time execution metrics and Order Flow Imbalance tracking.

### Features

- `OrderBook` — bids/asks as sorted level maps; O(log n) insert/cancel
- `OFITracker` — `snapshot()` → raw OFI = ΔBid − ΔAsk; `normalised_ofi()` ∈ [−1, 1]
- `ExecutionMetrics` — fill rate, avg fill price, market impact bps, queue position

```python
from lob_simulator.core.book import OrderBook
from lob_simulator.core.order import Order, Side
from lob_simulator.metrics.ofi import OFITracker

book = OrderBook()
ofi  = OFITracker(book)

book.add_order(Order(order_id=1, side=Side.BUY,  price=99.0, quantity=500))
book.add_order(Order(order_id=2, side=Side.SELL, price=101.0, quantity=300))

ofi.snapshot()   # baseline
book.add_order(Order(order_id=3, side=Side.BUY, price=99.0, quantity=200))
print(ofi.snapshot())          # positive OFI → buy pressure
print(ofi.normalised_ofi())    # ∈ [-1, 1]
```

---

## CI

GitHub Actions runs the full test suite on every push and pull request.

```
pytest backtester/tests/ lob_simulator/tests/ -v
```

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Tech Stack

**Backend:** Python 3.11+, Polars, FastAPI, Uvicorn, yfinance, pyarrow
**Frontend:** React 18, TypeScript, Vite 5, Tailwind CSS 3, Recharts 2
**Testing:** pytest, pytest-cov
