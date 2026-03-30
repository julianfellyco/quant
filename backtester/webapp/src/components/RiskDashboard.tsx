import type { AlphaDecomposition, BenchmarkStats } from '../types'

interface Props {
  benchmark?: BenchmarkStats | null
  alphaDecomp?: AlphaDecomposition | null
}

function fmt(v: number | null | undefined, decimals = 3) {
  return v == null ? '—' : v.toFixed(decimals)
}
function pct(v: number | null | undefined) {
  return v == null ? '—' : `${(v * 100).toFixed(1)}%`
}

export default function RiskDashboard({ benchmark, alphaDecomp }: Props) {
  if (!benchmark && !alphaDecomp) return null

  return (
    <div className="card space-y-4">
      <p className="section-title">Benchmark & Alpha Decomposition</p>

      {benchmark && (
        <div>
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">Benchmark ({benchmark.ticker})</p>
          <div className="grid grid-cols-2 gap-2">
            <div className="metric-box">
              <span className="metric-label">Benchmark Return</span>
              <span className={`text-lg font-bold ${(benchmark.total_return ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {pct(benchmark.total_return)}
              </span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Benchmark Sharpe</span>
              <span className={`text-lg font-bold ${(benchmark.sharpe ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {fmt(benchmark.sharpe)}
              </span>
            </div>
          </div>
        </div>
      )}

      {alphaDecomp && (
        <div>
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">Alpha Decomposition</p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {[
              { label: 'Alpha (Ann.)',        value: pct(alphaDecomp.alpha_annual),           highlight: (alphaDecomp.alpha_annual ?? 0) > 0 },
              { label: 'Beta',               value: fmt(alphaDecomp.beta),                   highlight: null },
              { label: 'Info Ratio',         value: fmt(alphaDecomp.information_ratio),      highlight: (alphaDecomp.information_ratio ?? 0) > 0 },
              { label: 'Tracking Error',     value: pct(alphaDecomp.tracking_error),         highlight: null },
              { label: 'Up Capture',         value: fmt(alphaDecomp.up_capture),             highlight: (alphaDecomp.up_capture ?? 0) > 1 },
              { label: 'Down Capture',       value: fmt(alphaDecomp.down_capture),           highlight: (alphaDecomp.down_capture ?? 1) < 1 },
            ].map(m => (
              <div key={m.label} className="metric-box">
                <span className="metric-label">{m.label}</span>
                <span className={`text-base font-semibold ${
                  m.highlight === true ? 'text-emerald-400' :
                  m.highlight === false ? 'text-red-400' : 'text-gray-200'
                }`}>{m.value}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
