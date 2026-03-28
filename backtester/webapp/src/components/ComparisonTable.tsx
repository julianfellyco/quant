import type { MetricsSummary } from '../types'

function pct(v: number | null) {
  if (v == null) return '—'
  return `${(v * 100).toFixed(2)}%`
}

function two(v: number | null) {
  if (v == null) return '—'
  return v.toFixed(2)
}

function colorSharpe(v: number | null) {
  if (v == null) return 'text-gray-400'
  return v >= 1 ? 'text-emerald-400' : v >= 0 ? 'text-yellow-400' : 'text-red-400'
}

export default function ComparisonTable({ rows }: { rows: MetricsSummary[] }) {
  if (!rows.length) return null
  return (
    <div className="card overflow-x-auto">
      <p className="section-title">Strategy Comparison</p>
      <table className="data-table">
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Strategy</th>
            <th>Net Sharpe</th>
            <th>Sortino</th>
            <th>Return</th>
            <th className="hidden md:table-cell">MDD</th>
            <th className="hidden md:table-cell">Ann. Vol</th>
            <th className="hidden lg:table-cell">Trades</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td className="font-semibold text-brand-400">{r.ticker}</td>
              <td className="text-gray-300 capitalize">{r.strategy.replace('_', ' ')}</td>
              <td className={colorSharpe(r.net_sharpe)}>{two(r.net_sharpe)}</td>
              <td className={colorSharpe(r.sortino)}>{two(r.sortino)}</td>
              <td className={(r.total_return ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                {pct(r.total_return)}
              </td>
              <td className="hidden md:table-cell text-red-400">{pct(r.max_drawdown)}</td>
              <td className="hidden md:table-cell">{pct(r.annualised_vol)}</td>
              <td className="hidden lg:table-cell">{r.n_trades}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
