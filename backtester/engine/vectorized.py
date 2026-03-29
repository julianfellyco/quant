"""
engine/vectorized.py — VectorizedEngine: Polars-native P&L computation.

Point-in-Time rules enforced by this engine
-------------------------------------------
  Rule 1 — Signal lag:  the DataHandler shifts all signals by 1 bar before
            they arrive here. No signal is computed from the same bar's close.

  Rule 2 — Fill price:  execution uses close[t], the bar on which the lagged
            signal is first active.  (close[t−1] generated the signal;
            close[t] is the fill bar.)

  Rule 3 — Cost timing: transaction costs are charged at the same bar as the
            position change, using bar close as the execution price.

P&L pipeline (per bar t)
------------------------
  position[t]   = signal[t] × shares_per_unit
  trade[t]      = position[t] − position[t−1]
  gross_ret[t]  = signal[t] × log_return[t]
  cost_usd[t]   = f(trade[t], price[t], atr[t], volume[t], is_event[t])
  cost_frac[t]  = cost_usd[t] / initial_capital
  net_ret[t]    = gross_ret[t] − cost_frac[t]
  equity[t]     = initial_capital × exp(Σ net_ret[0..t])

Annualisation
-------------
For daily bars:  ann_factor = 252
For 1-min bars:  ann_factor = 252 × 390 = 98,280
Sharpe and Sortino multiply daily/per-bar std by √ann_factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from .costs import cost_for_ticker


# --------------------------------------------------------------------------- #
# Result container                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class BacktestResult:
    ticker:         str
    strategy_name:  str
    equity_curve:   pl.DataFrame   # date, equity, gross_log_ret, net_log_ret, cost_usd
    gross_sharpe:   float
    net_sharpe:     float
    sortino:        float
    max_drawdown:   float          # fraction, e.g. −0.15 = −15%
    total_return:   float          # arithmetic, e.g. 0.22 = +22%
    annualised_vol: float          # annualised std of net log returns
    total_cost_usd: float
    n_trades:       int

    def summary(self) -> str:
        w = 52
        bar = "─" * w
        return "\n".join([
            bar,
            f"  {self.ticker}  │  {self.strategy_name}",
            bar,
            f"  Net Sharpe       {self.net_sharpe:>+8.3f}",
            f"  Gross Sharpe     {self.gross_sharpe:>+8.3f}",
            f"  Sortino          {self.sortino:>+8.3f}",
            f"  Max Drawdown     {self.max_drawdown:>+8.2%}",
            f"  Total Return     {self.total_return:>+8.2%}",
            f"  Ann. Volatility  {self.annualised_vol:>8.2%}",
            f"  Total Cost       ${self.total_cost_usd:>10,.0f}",
            f"  # Trades         {self.n_trades:>8}",
            bar,
        ])


# --------------------------------------------------------------------------- #
# VectorizedEngine                                                              #
# --------------------------------------------------------------------------- #

class VectorizedEngine:
    """
    Vectorized backtesting engine.

    Parameters
    ----------
    initial_capital : float
        USD capital deployed (denominator for cost-as-fraction calculation).
    shares_per_unit : int
        Number of shares per signal unit (+1 / −1).  Set to 1 for fractional.
    risk_free_rate : float
        Annualised risk-free rate (e.g. 0.05 for 5%).  Used in Sharpe/Sortino.
    ann_factor : int
        Trading bars per year.  252 for daily, 98_280 for 1-minute.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        shares_per_unit: int   = 1_000,
        risk_free_rate:  float = 0.05,
        ann_factor:      int   = 252,
    ) -> None:
        self.initial_capital = initial_capital
        self.shares_per_unit = shares_per_unit
        self.risk_free_rate  = risk_free_rate
        self.ann_factor      = ann_factor

    # ------------------------------------------------------------------ #
    # Entry point                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        ticker:        str,
        aligned_df:    pl.DataFrame,
        strategy_name: str = "unnamed",
    ) -> BacktestResult:
        """
        Execute backtest on an aligned DataFrame produced by
        DataHandler.align_signals().

        Required columns in aligned_df:
            timestamp        Datetime or Date
            close            Float64
            log_return       Float64
            atr_14           Float64
            signal           Float64    {−1, 0, +1}, already shifted 1 bar
        Optional columns (used by cost model if present):
            volume           Int64
            avg_volume_20    Float64
            is_event_window  Boolean
        """
        self._validate(aligned_df)
        df = self._compute_pnl(ticker, aligned_df)

        rfr_per_bar = math.log(1 + self.risk_free_rate) / self.ann_factor
        gross_rets  = df["gross_log_ret"]
        net_rets    = df["net_log_ret"]

        return BacktestResult(
            ticker         = ticker,
            strategy_name  = strategy_name,
            equity_curve   = df.select(
                ["timestamp", "equity", "gross_log_ret", "net_log_ret",
                 "transaction_cost_usd"]
            ),
            gross_sharpe   = self._sharpe(gross_rets, rfr_per_bar),
            net_sharpe     = self._sharpe(net_rets,   rfr_per_bar),
            sortino        = self._sortino(net_rets,   rfr_per_bar),
            max_drawdown   = self._max_drawdown(df["equity"]),
            total_return   = float(df["equity"][-1]) / self.initial_capital - 1.0,
            annualised_vol = float(net_rets.std() or 0.0) * math.sqrt(self.ann_factor),
            total_cost_usd = float(df["transaction_cost_usd"].sum()),
            n_trades       = int((df["trade_size"].abs() > 0).sum()),
        )

    # ------------------------------------------------------------------ #
    # P&L computation                                                       #
    # ------------------------------------------------------------------ #

    def _compute_pnl(self, ticker: str, df: pl.DataFrame) -> pl.DataFrame:
        shares = float(self.shares_per_unit)

        df = df.with_columns(
            (pl.col("signal") * shares).alias("position_shares")
        ).with_columns(
            (pl.col("position_shares") - pl.col("position_shares").shift(1))
            .fill_null(0.0).alias("trade_size")
        ).with_columns(
            (pl.col("signal") * pl.col("log_return")).fill_null(0.0)
            .alias("gross_log_ret")
        )

        # Pull optional columns for the cost model
        is_event   = df["is_event_window"]  if "is_event_window"  in df.columns else None
        bar_vol    = df["volume"].cast(pl.Float64) if "volume"       in df.columns else None
        avg_vol    = df["avg_volume_20"]    if "avg_volume_20"    in df.columns else None

        cost_series = cost_for_ticker(
            ticker      = ticker,
            trade_sizes = df["trade_size"],
            prices      = df["close"],
            atr         = df["atr_14"].fill_null(df["atr_14"].median() or 0.0),
            is_event    = is_event,
            bar_volume  = bar_vol,
            avg_volume  = avg_vol,
            df          = df,
        )

        df = df.with_columns(cost_series).with_columns(
            (pl.col("transaction_cost_usd") / self.initial_capital)
            .alias("cost_fraction")
        ).with_columns(
            (pl.col("gross_log_ret") - pl.col("cost_fraction"))
            .alias("net_log_ret")
        ).with_columns(
            (pl.col("net_log_ret").cum_sum().exp() * self.initial_capital)
            .alias("equity")
        )

        return df

    # ------------------------------------------------------------------ #
    # Risk metrics — all computed manually                                  #
    # ------------------------------------------------------------------ #

    def _sharpe(self, log_rets: pl.Series, rfr_per_bar: float) -> float:
        """
        Annualised Sharpe Ratio.

            Sharpe = mean(r − rf) / std(r − rf) × √ann_factor

        Uses sample standard deviation (ddof = 1).
        Returns NaN on fewer than 5 bars or near-zero vol (prevents blowup
        on short OOS windows where std ≈ 0). Clamped to [-50, 50].
        """
        if len(log_rets) < 5:
            return float("nan")
        excess = log_rets - rfr_per_bar
        mu     = float(excess.mean() or 0.0)
        sigma  = float(excess.std()  or 0.0)
        if sigma < 1e-8:
            return float("nan")
        raw = (mu / sigma) * math.sqrt(self.ann_factor)
        return max(-50.0, min(50.0, raw))

    def _sortino(self, log_rets: pl.Series, rfr_per_bar: float) -> float:
        """
        Annualised Sortino Ratio.

            Sortino = mean(r − rf) / σ_downside × √ann_factor

        σ_downside = std of bars where (r − rf) < 0 only.

        Quant Why for pharma: NVO had large positive spikes on FDA approval
        and trial readouts.  Sharpe penalises these equally with negative
        spikes, understating risk-adjusted performance.  Sortino correctly
        ignores upside volatility that doesn't represent investor risk.
        """
        excess   = log_rets - rfr_per_bar
        mu       = float(excess.mean() or 0.0)
        downside = excess.filter(excess < 0.0)
        if len(downside) < 2:
            return float("nan")
        sigma_d = float(downside.std() or 1e-9)
        return (mu / sigma_d) * math.sqrt(self.ann_factor)

    def _max_drawdown(self, equity: pl.Series) -> float:
        """
        Maximum Drawdown.

            MDD = min_t ( equity[t] / max_{s≤t}(equity[s]) − 1 )

        Uses cumulative maximum (O(n)) rather than an O(n²) nested loop.

        Quant Why: MDD is the primary capital-at-risk metric.  A strategy
        with Sharpe 1.2 but MDD −40% will be killed by a risk manager before
        it recovers.  For pharma backtests during binary events, MDD is often
        more informative than Sharpe because a single FDA rejection can produce
        a −30% drawdown in a single bar — far outside the Gaussian Sharpe model.
        """
        running_max = equity.cum_max()
        drawdowns   = equity / running_max - 1.0
        return float(drawdowns.min() or 0.0)

    # ------------------------------------------------------------------ #
    # Validation                                                            #
    # ------------------------------------------------------------------ #

    def _validate(self, df: pl.DataFrame) -> None:
        required = {"timestamp", "close", "log_return", "atr_14", "signal"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"aligned_df missing columns: {missing}")
        if df["signal"].is_null().all():
            raise ValueError("signal column is all null — run DataHandler.align_signals() first.")
