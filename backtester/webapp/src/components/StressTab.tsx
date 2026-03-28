import { useState } from 'react'
import {
  ResponsiveContainer, BarChart, Bar,
  XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine,
} from 'recharts'
import type { TickerInfo, StressResponse } from '../types'
import { runStress } from '../api'

interface Props { tickers: TickerInfo[] }

export default function StressTab({ tickers }: Props) {
  const knownTickers = tickers.length ? tickers.map(t => t.ticker) : ['NVO', 'PFE']

  const [ticker,         setTicker]         = useState('PFE')
  const [strategy,       setStrategy]       = useState('momentum')
  const [startDate,      setStartDate]      = useState('2023-01-01')
  const [endDate,        setEndDate]        = useState('2024-12-31')
  const [granularity,    setGranularity]    = useState<'daily' | 'hour' | 'minute'>('daily')
  const [nSimulations,   setNSimulations]   = useState(500)
  const [maxShiftDays,   setMaxShiftDays]   = useState(5)
  const [seed,           setSeed]           = useState(42)
  const [initialCapital, setInitialCapital] = useState(100_000)
  const [sharesPerUnit,  setSharesPerUnit]  = useState(100)
  const [entryZ,         setEntryZ]         = useState(1.5)
  const [exitZ,          setExitZ]          = useState(0.3)
  const [useEventHedge,  setUseEventHedge]  = useState(false)

  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)
  const [result,  setResult]  = useState<StressResponse | null>(null)

  async function submit() {
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await runStress({
        ticker, strategy, start_date: startDate, end_date: endDate, granularity,
        n_simulations: nSimulations, max_shift_days: maxShiftDays, seed,
        initial_capital: initialCapital, shares_per_unit: sharesPerUnit,
        entry_z: entryZ, exit_z: exitZ, use_event_hedge: useEventHedge,
      })
      setResult(res)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  function two(v: number | null | undefined) { return v == null ? '—' : v.toFixed(2) }

  const fragility = result?.fragility_score ?? 0
  const fragColour = fragility < 0.1 ? 'text-emerald-400' : fragility < 0.3 ? 'text-yellow-400' : 'text-red-400'

  const histData = result?.histogram_bins.map(b => ({
    label: b.bin_start.toFixed(2),
    count: b.count,
  })) ?? []

  return (
    <div className="space-y-6">
      {/* ── Params ─────────────────────────────────────────── */}
      <div className="card space-y-5">
        <p className="section-title">Monte Carlo Event Shuffling</p>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div>
            <label className="label">Ticker</label>
            <select className="input" value={ticker} onChange={e => setTicker(e.target.value)}>
              {knownTickers.map(t => <option key={t}>{t}</option>)}
            </select>
          </div>
          <div>
            <label className="label">Strategy</label>
            <select className="input" value={strategy} onChange={e => setStrategy(e.target.value)}>
              <option value="momentum">Momentum</option>
              <option value="mean_reversion">Mean Reversion</option>
            </select>
          </div>
          <div>
            <label className="label">Start Date</label>
            <input type="date" className="input" value={startDate} onChange={e => setStartDate(e.target.value)} />
          </div>
          <div>
            <label className="label">End Date</label>
            <input type="date" className="input" value={endDate} onChange={e => setEndDate(e.target.value)} />
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div>
            <label className="label">Simulations</label>
            <input type="number" className="input" value={nSimulations}
              onChange={e => setNSimulations(Number(e.target.value))} min={50} max={2000} step={50} />
          </div>
          <div>
            <label className="label">Max Shift Days</label>
            <input type="number" className="input" value={maxShiftDays}
              onChange={e => setMaxShiftDays(Number(e.target.value))} min={1} max={30} />
          </div>
          <div>
            <label className="label">Seed</label>
            <input type="number" className="input" value={seed} onChange={e => setSeed(Number(e.target.value))} />
          </div>
          <div>
            <label className="label">Granularity</label>
            <select className="input" value={granularity} onChange={e => setGranularity(e.target.value as typeof granularity)}>
              <option value="daily">Daily</option>
              <option value="hour">Hourly</option>
              <option value="minute">Minute</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
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
          <div>
            <label className="label">Entry Z</label>
            <input type="number" className="input" value={entryZ}
              onChange={e => setEntryZ(Number(e.target.value))} step={0.1} />
          </div>
          <div>
            <label className="label">Exit Z</label>
            <input type="number" className="input" value={exitZ}
              onChange={e => setExitZ(Number(e.target.value))} step={0.1} />
          </div>
        </div>

        <div className="flex items-center justify-between flex-wrap gap-3">
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <div onClick={() => setUseEventHedge(v => !v)}
              className={`w-10 h-5 rounded-full relative transition-colors ${useEventHedge ? 'bg-brand-600' : 'bg-gray-700'}`}>
              <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${useEventHedge ? 'translate-x-5' : ''}`} />
            </div>
            <span className="text-sm text-gray-300">Event Hedge</span>
          </label>
          <button className="btn-primary" onClick={submit} disabled={loading}>
            {loading ? <><span className="spinner" /> Running…</> : 'Run Stress Test'}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <div className="space-y-4">
          {/* KPI row */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {[
              { label: 'Base Sharpe',     value: two(result.base_sharpe),     },
              { label: 'Base Return',     value: result.base_return != null ? `${(result.base_return * 100).toFixed(2)}%` : '—' },
              { label: 'Base MDD',        value: result.base_mdd    != null ? `${(result.base_mdd    * 100).toFixed(2)}%` : '—' },
              { label: 'P5 Sharpe',       value: two(result.p5_sharpe)        },
              { label: 'P95 Sharpe',      value: two(result.p95_sharpe)       },
              { label: 'Worst Sharpe',    value: two(result.worst_sharpe)     },
              { label: 'Best Sharpe',     value: two(result.best_sharpe)      },
              { label: 'Simulations',     value: String(result.n_simulations) },
            ].map(m => (
              <div key={m.label} className="metric-box">
                <span className="metric-label">{m.label}</span>
                <span className="metric-value">{m.value}</span>
              </div>
            ))}

            <div className="metric-box">
              <span className="metric-label">Fragility Score</span>
              <span className={`text-lg font-bold ${fragColour}`}>
                {(fragility * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-gray-600">P(Sharpe &lt; 0)</span>
            </div>
          </div>

          {/* Histogram */}
          {histData.length > 0 && (
            <div className="card">
              <p className="section-title">Sharpe Distribution ({result.n_simulations} sims, shift ±{result.max_shift_days} days)</p>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={histData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="label" tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} width={36} />
                  <Tooltip
                    contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8, fontSize: 11 }}
                    labelStyle={{ color: '#9ca3af' }}
                    formatter={(v: number) => [v, 'Count']}
                  />
                  <ReferenceLine x={two(result.base_sharpe)} stroke="#f59e0b" strokeDasharray="4 2" label={{ value: 'Base', fill: '#f59e0b', fontSize: 10 }} />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
