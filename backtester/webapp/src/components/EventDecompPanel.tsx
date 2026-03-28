import type { SingleBacktestResult } from '../types'

function pct(v: number | null) {
  if (v == null) return '—'
  return `${(v * 100).toFixed(2)}%`
}

function two(v: number | null) {
  if (v == null) return '—'
  return v.toFixed(2)
}

export default function EventDecompPanel({ result }: { result: SingleBacktestResult }) {
  const { full, event, non_event } = result.event_decomp
  const rows = [
    { label: 'Full period',   w: full      },
    { label: 'Event window',  w: event     },
    { label: 'Non-event',     w: non_event },
  ]

  return (
    <div className="card overflow-x-auto">
      <p className="section-title">Event Decomposition</p>
      <table className="data-table">
        <thead>
          <tr>
            <th>Period</th>
            <th>Days</th>
            <th>Log Ret</th>
            <th>Sharpe</th>
            <th>MDD</th>
            <th className="hidden sm:table-cell">% of Total</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.label}>
              <td className="font-medium text-gray-200">{r.label}</td>
              <td>{r.w.n_days}</td>
              <td className={(r.w.total_log_ret ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                {pct(r.w.total_log_ret)}
              </td>
              <td className={(r.w.ann_sharpe ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                {two(r.w.ann_sharpe)}
              </td>
              <td className="text-red-400">{pct(r.w.max_drawdown)}</td>
              <td className="hidden sm:table-cell text-gray-400">{pct(r.w.pct_of_total)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
