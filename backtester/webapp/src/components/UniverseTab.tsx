import { useState } from 'react'
import {
  ResponsiveContainer, BarChart, Bar,
  XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine, Cell,
} from 'recharts'
import type { UniverseResponse } from '../types'
import { runUniverse } from '../api'

const PRESETS = [
  { id: 'pharma',    label: 'Pharma / Biotech',  desc: '25 large-cap pharma tickers' },
  { id: 'sp500',     label: 'S&P 500',            desc: 'Top 500 US stocks (Wikipedia)' },
  { id: 'nasdaq100', label: 'Nasdaq-100',          desc: 'Top 100 tech/growth stocks'   },
  { id: 'custom',    label: 'Custom',              desc: 'Enter your own tickers'       },
]

const STRATEGIES = [
  { id: 'momentum_rank',       label: 'Momentum Rank',       desc: 'Long top 20% / short bottom 20% by trailing return' },
  { id: 'mean_reversion_rank', label: 'Mean Reversion Rank', desc: 'Long most oversold / short most overbought by Z-score' },
  { id: 'momentum',            label: 'Momentum (single)',   desc: 'Classic momentum signal per ticker' },
  { id: 'mean_reversion',      label: 'Mean Rev (single)',   desc: 'Z-score mean reversion per ticker' },
]

export default function UniverseTab() {
  const [preset,         setPreset]         = useState<'pharma'|'sp500'|'nasdaq100'|'custom'>('pharma')
  const [customInput,    setCustomInput]     = useState('AAPL, MSFT, NVDA, GOOGL, AMZN')
  const [strategy,       setStrategy]       = useState<'momentum_rank'|'mean_reversion_rank'|'momentum'|'mean_reversion'>('momentum_rank')
  const [startDate,      setStartDate]      = useState('2024-01-01')
  const [endDate,        setEndDate]        = useState('2024-12-31')
  const [topPct,         setTopPct]         = useState(0.2)
  const [lookback,       setLookback]       = useState(60)
  const [maxTickers,     setMaxTickers]     = useState(25)
  const [initialCapital, setInitialCapital] = useState(100_000)
  const [sharesPerUnit,  setSharesPerUnit]  = useState(100)

  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)
  const [result,  setResult]  = useState<UniverseResponse | null>(null)

  function parseCustom() {
    return customInput.split(/[,\s]+/).map(t => t.trim().toUpperCase()).filter(Boolean)
  }

  async function submit() {
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await runUniverse({
        preset,
        custom_tickers:  preset === 'custom' ? parseCustom() : [],
        strategy,
        start_date:      startDate,
        end_date:        endDate,
        top_pct:         topPct,
        bottom_pct:      topPct,
        lookback,
        initial_capital: initialCapital,
        shares_per_unit: sharesPerUnit,
        risk_free_rate:  0.05,
        max_tickers:     maxTickers,
      })
      setResult(res)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  function two(v: number | null | undefined) { return v == null ? '—' : v.toFixed(2) }
  function pct(v: number | null | undefined) { return v == null ? '—' : `${(v * 100).toFixed(1)}%` }

  // Bar chart: top 20 by sharpe
  const chartData = result
    ? result.leaderboard.slice(0, 20).map(r => ({
        ticker: r.ticker,
        sharpe: r.net_sharpe ?? 0,
      }))
    : []

  return (
    <div className="space-y-6">
      {/* ── Params ─────────────────────────────────────────── */}
      <div className="card space-y-5">
        <p className="section-title">Universe Backtest</p>

        {/* Preset selector */}
        <div>
          <p className="label">Universe</p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {PRESETS.map(p => (
              <button
                key={p.id}
                onClick={() => setPreset(p.id as typeof preset)}
                className={`rounded-lg border px-3 py-2 text-left transition-colors
                  ${preset === p.id
                    ? 'bg-brand-600 border-brand-500 text-white'
                    : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600'}`}
              >
                <p className="text-xs font-semibold">{p.label}</p>
                <p className="text-xs opacity-60 mt-0.5">{p.desc}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Custom tickers input */}
        {preset === 'custom' && (
          <div>
            <label className="label">Tickers <span className="text-gray-600 font-normal">(comma or space separated)</span></label>
            <input
              className="input"
              placeholder="AAPL, MSFT, NVDA, GOOGL"
              value={customInput}
              onChange={e => setCustomInput(e.target.value)}
            />
            {parseCustom().length > 0 && (
              <div className="flex flex-wrap gap-1.5 mt-2">
                {parseCustom().map(t => (
                  <span key={t} className="px-2 py-0.5 rounded-full text-xs font-semibold bg-brand-600/20 border border-brand-600/40 text-brand-400">{t}</span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Strategy selector */}
        <div>
          <p className="label">Strategy</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {STRATEGIES.map(s => (
              <button
                key={s.id}
                onClick={() => setStrategy(s.id as typeof strategy)}
                className={`rounded-lg border px-3 py-2 text-left transition-colors
                  ${strategy === s.id
                    ? 'bg-brand-600 border-brand-500 text-white'
                    : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600'}`}
              >
                <p className="text-xs font-semibold">{s.label}</p>
                <p className="text-xs opacity-60 mt-0.5">{s.desc}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Date + params */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div>
            <label className="label">Start Date</label>
            <input type="date" className="input" value={startDate} onChange={e => setStartDate(e.target.value)} />
          </div>
          <div>
            <label className="label">End Date</label>
            <input type="date" className="input" value={endDate} onChange={e => setEndDate(e.target.value)} />
          </div>
          <div>
            <label className="label">Max Tickers</label>
            <input type="number" className="input" value={maxTickers}
              onChange={e => setMaxTickers(Number(e.target.value))} min={2} max={500} />
          </div>
          <div>
            <label className="label">Lookback (bars)</label>
            <input type="number" className="input" value={lookback}
              onChange={e => setLookback(Number(e.target.value))} min={10} max={252} />
          </div>
          <div>
            <label className="label">Long/Short %</label>
            <input type="number" className="input" value={topPct}
              onChange={e => setTopPct(Number(e.target.value))} step={0.05} min={0.05} max={0.5} />
          </div>
          <div>
            <label className="label">Initial Capital</label>
            <input type="number" className="input" value={initialCapital}
              onChange={e => setInitialCapital(Number(e.target.value))} min={1000} step={10000} />
          </div>
          <div>
            <label className="label">Shares / Unit</label>
            <input type="number" className="input" value={sharesPerUnit}
              onChange={e => setSharesPerUnit(Number(e.target.value))} min={1} />
          </div>
        </div>

        <div className="flex justify-end">
          <button className="btn-primary" onClick={submit} disabled={loading}>
            {loading ? <><span className="spinner" /> Running universe…</> : 'Run Universe Backtest'}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {/* ── Results ─────────────────────────────────────────── */}
      {result && (
        <div className="space-y-4">
          {/* KPI row */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <div className="metric-box">
              <span className="metric-label">Tickers Run</span>
              <span className="metric-value">{result.tickers_run}</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Mean Sharpe</span>
              <span className={`text-lg font-bold ${(result.mean_sharpe ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {two(result.mean_sharpe)}
              </span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Median Sharpe</span>
              <span className={`text-lg font-bold ${(result.median_sharpe ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {two(result.median_sharpe)}
              </span>
            </div>
            <div className="metric-box">
              <span className="metric-label">% Positive Sharpe</span>
              <span className="metric-value">{pct(result.pct_positive_sharpe)}</span>
            </div>
            {result.best_ticker && (
              <div className="metric-box">
                <span className="metric-label">Best Ticker</span>
                <span className="text-lg font-bold text-emerald-400">{result.best_ticker}</span>
              </div>
            )}
            {result.worst_ticker && (
              <div className="metric-box">
                <span className="metric-label">Worst Ticker</span>
                <span className="text-lg font-bold text-red-400">{result.worst_ticker}</span>
              </div>
            )}
          </div>

          {/* Sharpe bar chart */}
          {chartData.length > 0 && (
            <div className="card">
              <p className="section-title">Net Sharpe — Top {chartData.length} Tickers</p>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="ticker" tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} width={36} />
                  <Tooltip
                    contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8, fontSize: 11 }}
                    formatter={(v: number) => [v.toFixed(3), 'Net Sharpe']}
                  />
                  <ReferenceLine y={0} stroke="#374151" />
                  <Bar dataKey="sharpe" radius={[3, 3, 0, 0]}>
                    {chartData.map((d, i) => (
                      <Cell key={i} fill={d.sharpe >= 0 ? '#10b981' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Leaderboard table */}
          <div className="card overflow-x-auto">
            <p className="section-title">Full Leaderboard — {result.strategy.replace('_', ' ')}</p>
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Ticker</th>
                  <th>Net Sharpe</th>
                  <th>Sortino</th>
                  <th>Return</th>
                  <th className="hidden md:table-cell">MDD</th>
                  <th className="hidden md:table-cell">Trades</th>
                  <th className="hidden lg:table-cell">Costs</th>
                </tr>
              </thead>
              <tbody>
                {result.leaderboard.map((r, i) => (
                  <tr key={r.ticker}>
                    <td className="text-gray-600">{i + 1}</td>
                    <td className="font-semibold text-brand-400">{r.ticker}</td>
                    <td className={(r.net_sharpe ?? 0) >= 0 ? 'text-emerald-400 font-semibold' : 'text-red-400 font-semibold'}>
                      {two(r.net_sharpe)}
                    </td>
                    <td className={(r.sortino ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                      {two(r.sortino)}
                    </td>
                    <td className={(r.total_return ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                      {pct(r.total_return)}
                    </td>
                    <td className="hidden md:table-cell text-red-400">{pct(r.max_drawdown)}</td>
                    <td className="hidden md:table-cell">{r.n_trades}</td>
                    <td className="hidden lg:table-cell text-gray-500">
                      ${r.total_cost_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
