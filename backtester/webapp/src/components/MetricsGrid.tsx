import type { MetricsSummary } from '../types'

function pct(v: number | null) {
  if (v == null) return '—'
  return `${(v * 100).toFixed(2)}%`
}

function two(v: number | null) {
  if (v == null) return '—'
  return v.toFixed(2)
}

function usd(v: number) {
  return v >= 1_000 ? `$${(v / 1_000).toFixed(1)}k` : `$${v.toFixed(0)}`
}

interface Metric { label: string; value: string; pos?: boolean; neg?: boolean }

function Box({ label, value, pos, neg }: Metric) {
  const cls = pos
    ? 'metric-value-pos'
    : neg
    ? 'metric-value-neg'
    : 'metric-value'
  return (
    <div className="metric-box">
      <span className="metric-label">{label}</span>
      <span className={cls}>{value}</span>
    </div>
  )
}

export default function MetricsGrid({ m }: { m: MetricsSummary }) {
  const ret      = m.total_return ?? 0
  const mdd      = m.max_drawdown ?? 0
  const sharpe   = m.net_sharpe ?? 0

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
      <Box label="Gross Sharpe"  value={two(m.gross_sharpe)} pos={( m.gross_sharpe  ?? 0) > 0} neg={(m.gross_sharpe  ?? 0) < 0} />
      <Box label="Net Sharpe"    value={two(m.net_sharpe)}   pos={sharpe > 0}                   neg={sharpe < 0}                  />
      <Box label="Sortino"       value={two(m.sortino)}      pos={(m.sortino        ?? 0) > 0} neg={(m.sortino        ?? 0) < 0} />
      <Box label="Total Return"  value={pct(m.total_return)} pos={ret > 0}                      neg={ret < 0}                     />
      <Box label="Ann. Vol"      value={pct(m.annualised_vol)}                                                                    />
      <Box label="Max Drawdown"  value={pct(m.max_drawdown)} neg={mdd < -0.05}                                                    />
      <Box label="# Trades"      value={String(m.n_trades)}                                                                       />
      <Box label="Total Costs"   value={usd(m.total_cost_usd)}                                                                    />
    </div>
  )
}
