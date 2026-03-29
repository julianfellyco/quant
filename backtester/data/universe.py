"""backtester/data/universe.py — Pre-defined ticker universes and sector mappings."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Universe:
    """A named collection of tickers with optional sector mappings."""

    name: str
    tickers: list[str]
    sectors: dict[str, list[str]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tickers)


# Pre-defined universes
PHARMA = Universe(
    name="pharma_majors",
    tickers=["PFE", "NVO", "LLY", "JNJ", "MRK", "ABBV", "BMY", "AZN", "AMGN", "GILD"],
    sectors={
        "pharma": ["PFE", "NVO", "LLY", "JNJ", "MRK", "ABBV", "BMY", "AZN", "AMGN", "GILD"],
    },
)

TECH = Universe(
    name="tech_megacaps",
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSM", "AVGO", "ORCL", "CRM"],
    sectors={
        "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSM", "AVGO", "ORCL", "CRM"],
    },
)

ENERGY = Universe(
    name="energy",
    tickers=["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HAL"],
    sectors={
        "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HAL"],
    },
)

UNIVERSES: dict[str, Universe] = {
    "pharma":  PHARMA,
    "tech":    TECH,
    "energy":  ENERGY,
}
