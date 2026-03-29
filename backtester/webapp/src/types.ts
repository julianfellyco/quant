// types.ts — TypeScript interfaces mirroring all Pydantic response models

export type Strategy    = 'momentum' | 'mean_reversion'
export type Granularity = 'daily' | 'hour' | 'minute'
export type PairState   = 'flat' | 'long_spread' | 'short_spread'
export type Tab         = 'backtest' | 'pairs' | 'stress' | 'walkforward' | 'universe'

// ── /api/tickers ──────────────────────────────────────────────────────────
export interface BinaryEvent {
  date:        string
  event_type:  'FDA_DECISION' | 'CLINICAL_TRIAL' | 'EARNINGS' | 'CONFERENCE' | 'MACRO'
  description: string
  pre_window:  number
  post_window: number
}

export interface TickerInfo {
  ticker:            string
  name:              string
  exchange:          string
  avg_daily_volume:  number
  approx_price_2024: number
  base_bps:          number
  events:            BinaryEvent[]
}

export interface TickersResponse {
  tickers: TickerInfo[]
}

// ── /api/backtest ─────────────────────────────────────────────────────────
export interface BacktestRequest {
  tickers:         string[]
  start_date:      string
  end_date:        string
  strategies:      string[]
  granularity:     Granularity
  initial_capital: number
  shares_per_unit: number
  risk_free_rate:  number
  use_event_hedge: boolean
  entry_z:         number
  exit_z:          number
}

export interface EquityCurvePoint {
  timestamp:     string
  equity:        number
  gross_log_ret: number
  net_log_ret:   number
  cost_usd:      number
}

export interface MetricsSummary {
  ticker:         string
  strategy:       string
  gross_sharpe:   number | null
  net_sharpe:     number | null
  sortino:        number | null
  max_drawdown:   number | null
  total_return:   number | null
  annualised_vol: number | null
  total_cost_usd: number
  n_trades:       number
}

export interface WindowStats {
  label:         string
  n_days:        number
  total_log_ret: number
  ann_sharpe:    number | null
  max_drawdown:  number | null
  pct_of_total:  number | null
}

export interface SingleBacktestResult {
  metrics:      MetricsSummary
  equity_curve: EquityCurvePoint[]
  event_decomp: {
    full:      WindowStats
    event:     WindowStats
    non_event: WindowStats
  }
}

export interface BacktestResponse {
  results:          SingleBacktestResult[]
  comparison_table: MetricsSummary[]
}

// ── /api/pairs ────────────────────────────────────────────────────────────
export interface PairsRequest {
  start_date:      string
  end_date:        string
  granularity:     Granularity
  hedge_window:    number
  zscore_window:   number
  entry_z:         number
  exit_z:          number
  use_event_hedge: boolean
}

export interface SpreadPoint {
  timestamp:   string
  nvo_close:   number
  pfe_close:   number
  beta:        number | null
  spread:      number | null
  spread_z:    number | null
  nvo_signal:  number | null
  pfe_signal:  number | null
  pair_state:  PairState | null
  spread_mean: number | null
  spread_std:  number | null
}

export interface PairsSummaryOut {
  mean_beta:        number | null
  std_beta:         number | null
  spread_z_mean:    number | null
  spread_z_std:     number | null
  pct_long_spread:  number | null
  pct_short_spread: number | null
  pct_flat:         number | null
}

export interface PairsResponse {
  spread_data: SpreadPoint[]
  summary:     PairsSummaryOut
}

// ── /api/stress ───────────────────────────────────────────────────────────
export interface StressRequest {
  ticker:          string
  strategy:        string
  start_date:      string
  end_date:        string
  granularity:     Granularity
  n_simulations:   number
  max_shift_days:  number
  seed:            number
  initial_capital: number
  shares_per_unit: number
  entry_z:         number
  exit_z:          number
  use_event_hedge: boolean
}

export interface HistogramBin {
  bin_start: number
  bin_end:   number
  count:     number
}

export interface StressResponse {
  base_sharpe:         number | null
  base_return:         number | null
  base_mdd:            number | null
  p5_sharpe:           number | null
  p95_sharpe:          number | null
  fragility_score:     number
  worst_sharpe:        number | null
  best_sharpe:         number | null
  n_simulations:       number
  max_shift_days:      number
  histogram_bins:      HistogramBin[]
  sharpe_distribution: (number | null)[]
}

// ── /api/walkforward ─────────────────────────────────────────────────────
export interface WalkForwardRequest {
  ticker:          string
  strategy:        string
  start_date:      string
  end_date:        string
  granularity:     Granularity
  train_bars:      number
  test_bars:       number
  step_bars:       number | null
  optimise_on:     string
  initial_capital: number
  shares_per_unit: number
  param_grid:      Record<string, number[]>
}

export interface FoldOut {
  fold_index:  number
  train_start: string
  train_end:   string
  test_start:  string
  test_end:    string
  best_params: Record<string, number>
  is_sharpe:   number | null
  oos_sharpe:  number | null
  oos_return:  number | null
  oos_mdd:     number | null
}

export interface WalkForwardResponse {
  folds:                FoldOut[]
  aggregate_oos_sharpe: number | null
  is_sharpe_mean:       number | null
  stability_score:      number | null
  sharpe_degradation:   number | null
  n_folds:              number
}

// ── /api/universe ─────────────────────────────────────────────────────────
export interface UniverseRequest {
  preset:          'pharma' | 'sp500' | 'nasdaq100' | 'custom'
  custom_tickers:  string[]
  strategy:        'momentum_rank' | 'mean_reversion_rank' | 'momentum' | 'mean_reversion'
  start_date:      string
  end_date:        string
  top_pct:         number
  bottom_pct:      number
  lookback:        number
  initial_capital: number
  shares_per_unit: number
  risk_free_rate:  number
  max_tickers:     number
}

export interface TickerResult {
  ticker:         string
  net_sharpe:     number | null
  sortino:        number | null
  total_return:   number | null
  max_drawdown:   number | null
  n_trades:       number
  total_cost_usd: number
}

export interface UniverseResponse {
  tickers_run:          number
  strategy:             string
  leaderboard:          TickerResult[]
  mean_sharpe:          number | null
  median_sharpe:        number | null
  pct_positive_sharpe:  number | null
  best_ticker:          string | null
  worst_ticker:         string | null
}
