import { useState } from 'react'
import {
  ResponsiveContainer, ComposedChart, Line, Bar,
  XAxis, YAxis, Tooltip, CartesianGrid, Legend, ReferenceLine,
} from 'recharts'
import type { PairsResponse } from '../types'
import { runPairs } from '../api'

export default function PairsTab() {
  const [startDate,     setStartDate]     = useState('2023-01-01')
  const [endDate,       setEndDate]       = useState('2024-12-31')
  const [granularity,   setGranularity]   = useState<'daily' | 'hour' | 'minute'>('daily')
  const [hedgeWindow,   setHedgeWindow]   = useState(60)
  const [zscoreWindow,  setZscoreWindow]  = useState(20)
  const [entryZ,        setEntryZ]        = useState(1.5)
  const [exitZ,         setExitZ]         = useState(0.3)
  const [useEventHedge, setUseEventHedge] = useState(false)

  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)
  const [result,  setResult]  = useState<PairsResponse | null>(null)

  async function submit() {
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await runPairs({
        start_date: startDate, end_date: endDate, granularity,
        hedge_window: hedgeWindow, zscore_window: zscoreWindow,
        entry_z: entryZ, exit_z: exitZ, use_event_hedge: useEventHedge,
      })
      setResult(res)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  function two(v: number | null | undefined) {
    return v == null ? '—' : v.toFixed(3)
  }
  function pct(v: number | null | undefined) {
    return v == null ? '—' : `${(v * 100).toFixed(1)}%`
  }

  // Downsample spread data for chart
  const chartData = result
    ? (() => {
        const step = Math.max(1, Math.floor(result.spread_data.length / 500))
        return result.spread_data
          .filter((_, i) => i % step === 0)
          .map(d => ({
            t:        d.timestamp.slice(0, 10),
            z:        d.spread_z,
            beta:     d.beta,
            state:    d.pair_state,
          }))
      })()
    : []

  const stateColor = (state: string | null | undefined) => {
    if (state === 'long_spread')  return '#10b981'
    if (state === 'short_spread') return '#ef4444'
    return '#374151'
  }

  return (
    <div className="space-y-6">
      {/* ── Params ─────────────────────────────────────────── */}
      <div className="card space-y-5">
        <p className="section-title">NVO / PFE Cointegration</p>

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
            <label className="label">Granularity</label>
            <select className="input" value={granularity} onChange={e => setGranularity(e.target.value as typeof granularity)}>
              <option value="daily">Daily</option>
              <option value="hour">Hourly</option>
              <option value="minute">Minute</option>
            </select>
          </div>
          <div>
            <label className="label">Hedge Window</label>
            <input type="number" className="input" value={hedgeWindow} onChange={e => setHedgeWindow(Number(e.target.value))} min={10} />
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div>
            <label className="label">Z-Score Window</label>
            <input type="number" className="input" value={zscoreWindow} onChange={e => setZscoreWindow(Number(e.target.value))} min={5} />
          </div>
          <div>
            <label className="label">Entry Z</label>
            <input type="number" className="input" value={entryZ} onChange={e => setEntryZ(Number(e.target.value))} step={0.1} />
          </div>
          <div>
            <label className="label">Exit Z</label>
            <input type="number" className="input" value={exitZ} onChange={e => setExitZ(Number(e.target.value))} step={0.1} />
          </div>
          <div className="flex items-end pb-1">
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <div onClick={() => setUseEventHedge(v => !v)}
                className={`w-10 h-5 rounded-full relative transition-colors ${useEventHedge ? 'bg-brand-600' : 'bg-gray-700'}`}>
                <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${useEventHedge ? 'translate-x-5' : ''}`} />
              </div>
              <span className="text-sm text-gray-300">Event Hedge</span>
            </label>
          </div>
        </div>

        <div className="flex justify-end">
          <button className="btn-primary" onClick={submit} disabled={loading}>
            {loading ? <><span className="spinner" /> Running…</> : 'Run Pairs Analysis'}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {/* ── Results ─────────────────────────────────────────── */}
      {result && (
        <div className="space-y-4">
          {/* Summary stats */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {[
              { label: 'Mean Beta',        value: two(result.summary.mean_beta)         },
              { label: 'Std Beta',         value: two(result.summary.std_beta)          },
              { label: 'Z-Score Mean',     value: two(result.summary.spread_z_mean)     },
              { label: 'Z-Score Std',      value: two(result.summary.spread_z_std)      },
              { label: '% Long Spread',    value: pct(result.summary.pct_long_spread)   },
              { label: '% Short Spread',   value: pct(result.summary.pct_short_spread)  },
              { label: '% Flat',           value: pct(result.summary.pct_flat)          },
            ].map(m => (
              <div key={m.label} className="metric-box">
                <span className="metric-label">{m.label}</span>
                <span className="metric-value">{m.value}</span>
              </div>
            ))}
          </div>

          {/* Spread Z-score chart */}
          <div className="card">
            <p className="section-title">Spread Z-Score &amp; Regime</p>
            <ResponsiveContainer width="100%" height={260}>
              <ComposedChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="t" tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                <YAxis yAxisId="z" tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} width={36} />
                <YAxis yAxisId="beta" orientation="right" tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} width={40} />
                <Tooltip
                  contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8, fontSize: 11 }}
                  labelStyle={{ color: '#9ca3af' }}
                />
                <Legend wrapperStyle={{ fontSize: 11, color: '#9ca3af' }} />
                <ReferenceLine yAxisId="z" y={entryZ}  stroke="#10b981" strokeDasharray="4 2" strokeOpacity={0.6} />
                <ReferenceLine yAxisId="z" y={-entryZ} stroke="#ef4444" strokeDasharray="4 2" strokeOpacity={0.6} />
                <ReferenceLine yAxisId="z" y={0}       stroke="#374151" strokeWidth={1} />
                <Bar yAxisId="z" dataKey="z" name="Spread Z" fill="#8b5cf6" opacity={0.6} maxBarSize={4}
                  label={false}
                />
                <Line yAxisId="beta" type="monotone" dataKey="beta" name="Rolling β"
                  stroke="#f59e0b" strokeWidth={1.5} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>

            {/* State legend */}
            <div className="flex gap-4 mt-2 text-xs">
              {[['long_spread', '#10b981', 'Long Spread'], ['short_spread', '#ef4444', 'Short Spread'], ['flat', '#374151', 'Flat']].map(([, color, label]) => (
                <span key={label} className="flex items-center gap-1.5 text-gray-400">
                  <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: color }} />
                  {label}
                </span>
              ))}
            </div>
          </div>

          {/* State timeline (coloured strip) */}
          <div className="card">
            <p className="section-title">Pair State Timeline</p>
            <div className="flex h-8 rounded overflow-hidden gap-px">
              {chartData.map((d, i) => (
                <div key={i} className="flex-1" style={{ background: stateColor(d.state) }} title={`${d.t}: ${d.state}`} />
              ))}
            </div>
            <div className="flex gap-4 mt-2 text-xs">
              {[['#10b981', 'Long Spread'], ['#ef4444', 'Short Spread'], ['#374151', 'Flat']].map(([color, label]) => (
                <span key={label} className="flex items-center gap-1.5 text-gray-400">
                  <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: color }} />
                  {label}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
