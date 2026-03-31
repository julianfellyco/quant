# Contributing

## Setup

```bash
git clone https://github.com/julianfellyco/quant.git
cd quant
make install
```

## Development workflow

1. Create a feature branch from `main`
2. Write tests first (pytest)
3. Implement the feature
4. Run `make lint` (ruff + mypy) and `make test`
5. Open a PR with a description of the changes

## Code style

- Python 3.11+, `from __future__ import annotations` at the top of every file
- Type hints on every function signature
- Polars over pandas — do not introduce pandas in new code
- Google-style docstrings on all public classes and functions
- 100-character line length (ruff enforced)
- Conventional commit messages: `feat(module): description`

## Testing

```bash
make test                                    # all tests
pytest backtester/tests/ -v                  # backtester only
pytest lob_simulator/tests/ -v               # LOB only
pytest -k "test_risk" -v                     # filter by name
pytest --cov=backtester --cov-report=html    # coverage report
```

All new features must include tests. Existing test coverage must not regress.

## Adding a new strategy

1. Add signal function to `backtester/strategy/signals.py`
2. Register in `STRATEGY_REGIME_MAP` in `backtester/strategy/regime.py`
3. Add test in `backtester/tests/`
4. Wire into `POST /api/backtest` if relevant

## Adding a new API endpoint

Follow the pattern in `backtester/api/routes/backtest.py`:

```python
from fastapi import APIRouter
router = APIRouter()

@router.post("/my-endpoint")
async def my_endpoint(req: MyRequest) -> MyResponse:
    ...
```

Then register in `backtester/api/main.py`:

```python
from backtester.api.routes.my_module import router as my_router
app.include_router(my_router, prefix="/api")
```
