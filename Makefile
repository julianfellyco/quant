.PHONY: install test lint fmt api dev build

install:
	pip install -e ".[dev]"

test:
	pytest -v --tb=short --cov=backtester --cov=lob_simulator

lint:
	ruff check .
	mypy backtester/ lob_simulator/ --ignore-missing-imports

fmt:
	ruff format .

api:
	PYTHONPATH=. uvicorn backtester.api.main:app --port 8000 --reload

dev:
	cd backtester/webapp && npm run dev

build:
	cd backtester/webapp && npm run build
