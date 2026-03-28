import { useState } from 'react'
import {
  ResponsiveContainer, ComposedChart, Bar, Line,
  XAxis, YAxis, Tooltip, CartesianGrid, Legend,
} from 'recharts'
import type { TickerInfo, WalkForwardResponse } from '../types'
import { runWalkForward } from '../api'

interface Props { tickers: TickerInfo[] }

export default function WalkForwardTab({ tickers }: Props) {
  const knownTickers = tickers.length ? tickers.map(t => t.ticker) : ['NVO', 'PFE']

  const [ticker,         setTicker]         = useState('PFE')
  const [strategy,       setStrategy]       = useState('momentum')
  const [startDate,      setStartDate]      = useState('2022-01-01')
  const [endDate,        setEndDate]        = useState('2024-12-31')
  const [granularity,    setGranularity]    = useState<'daily' | 'hour' | 'minute'>('daily')
  const [trainBars,      setTrainBars]      = useState(120)
  const [testBars,       setTestBars]       = useState(21)
  const [optimiseOn,     setOptimiseOn]     = useState('sharpe')
  const [initialCapital, setInitialCapital] = useState(100_000)
  const [sharesPerUnit,  setSharesPerUnit]  = useState(100)

  // Param grid
  const [entryZVals,  setEntryZVals]  = useState('1.0,1.5,2.0')
  const [exitZVals,   setExitZVals]   = useState('0.2,0.5')

  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)
  const [result,  setResult]  = useState<WalkForwardResponse | null>(null)

  function parseFloatList(s: string): number[] {
    return s.split(',').map(x => parseFloat(x.trim())).filter(x => !isNaN(x))
  }

  async function submit() {
    setLoading(true); setError(null); setResult(null)
    try {
      const paramGrid: Record<string, number[]> = {
        entry_z: parseFloatList(entryZVals),
        exit_z:  parseFloatList(exitZVals),
      }
      const res = await runWalkForward({
        ticker, strategy, start_date: startDate, end_date: endDate, granularity,
        train_bars: trainBars, test_bars: testBars, step_bars: null,
        optimise_on: optimiseOn, initial_capital: initialCapital,
        shares_per_unit: sharesPerUnit, param_grid: paramGrid,
      })
      setResult(res)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  function two(v: number | null | undefined) { return v == null ? '—' : v.toFixed(3) }

  const stabilityColor = (v: number | null | undefined) => {
    if (v == null) return 'metric-value'
    return v < 0.3 ? 'metric-value-pos' : v < 0.7 ? 'metric-value' : 'metric-value-neg'
  }

  const degradColor = (v: number | null | undefined) => {
    if (v == null) return 'metric-value'
    return v < 0.5 ? 'metric-value-pos' : v < 1.5 ? 'metric-value' : 'metric-value-neg'
  }

  const chartData = result?.folds.map(f => ({
    fold:       `F${f.fold_index + 1}`,
    is_sharpe:  f.is_sharpe,
    oos_sharpe: f.oos_sharpe,
    oos_return: f.oos_return != null ? f.oos_return * 100 : null,
  })) ?? []

  return (
    <div className="space-y-6">
      {/* ── Params ─────────────────────────────────────────── */}
      <div className="card space-y-5">
        <p className="section-title">Walk-Forward Optimiser</p>

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
            <label className="label">Train Bars</label>
            <input type="number" className="input" value={trainBars} onChange={e => setTrainBars(Number(e.target.value))} min={20} />
          </div>
          <div>
            <label className="label">Test Bars</label>
            <input type="number" className="input" value={testBars} onChange={e => setTestBars(Number(e.target.value))} min={5} />
          </div>
          <div>
            <label className="label">Optimise On</label>
            <select className="input" value={optimiseOn} onChange={e => setOptimiseOn(e.target.value)}>
              <option value="sharpe">Sharpe</option>
              <option value="return">Return</option>
              <option value="sortino">Sortino</option>
            </select>
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

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <label className="label">Entry Z values (comma-separated)</label>
            <input className="input" value={entryZVals} onChange={e => setEntryZVals(e.target.value)}
              placeholder="1.0,1.5,2.0" />
          </div>
          <div>
            <label className="label">Exit Z values (comma-separated)</label>
            <input className="input" value={exitZVals} onChange={e => setExitZVals(e.target.value)}
              placeholder="0.2,0.5" />
          </div>
        </div>

        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="grid grid-cols-2 gap-3 flex-1 max-w-xs">
            <div>
              <label className="label">Capital</label>
              <input type="number" className="input" value={initialCapital}
                onChange={e => setInitialCapital(Number(e.target.value))} min={1000} step={10000} />
            </div>
            <div>
              <label className="label">Shares / Unit</label>
              <input type="number" className="input" value={sharesPerUnit}
                onChange={e => setSharesPerUnit(Number(e.target.value))} min={1} />
            </div>
          </div>
          <button className="btn-primary" onClick={submit} disabled={loading}>
            {loading ? <><span className="spinner" /> Running…</> : 'Run Walk-Forward'}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <div className="space-y-4">
          {/* Aggregate KPIs */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <div className="metric-box">
              <span className="metric-label">Agg. OOS Sharpe</span>
              <span className={`text-lg font-bold ${(result.aggregate_oos_sharpe ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {two(result.aggregate_oos_sharpe)}
              </span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Mean IS Sharpe</span>
              <span className="metric-value">{two(result.is_sharpe_mean)}</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Stability Score</span>
              <span className={stabilityColor(result.stability_score)}>{two(result.stability_score)}</span>
              <span className="text-xs text-gray-600">std(OOS Sharpes)</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Sharpe Degradation</span>
              <span className={degradColor(result.sharpe_degradation)}>{two(result.sharpe_degradation)}</span>
              <span className="text-xs text-gray-600">IS − OOS gap</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Folds</span>
              <span className="metric-value">{result.n_folds}</span>
            </div>
          </div>

          {/* IS vs OOS Sharpe chart */}
          {chartData.length > 0 && (
            <div className="card">
              <p className="section-title">IS vs OOS Sharpe per Fold</p>
              <ResponsiveContainer width="100%" height={240}>
                <ComposedChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="fold" tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} />
                  <YAxis yAxisId="sharpe" tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} width={36} />
                  <YAxis yAxisId="ret" orientation="right" tick={{ fill: '#6b7280', fontSize: 10 }} tickLine={false} axisLine={false} width={40} tickFormatter={v => `${v.toFixed(0)}%`} />
                  <Tooltip
                    contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8, fontSize: 11 }}
                    labelStyle={{ color: '#9ca3af' }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11, color: '#9ca3af' }} />
                  <Bar yAxisId="sharpe" dataKey="is_sharpe"  name="IS Sharpe"  fill="#8b5cf6" opacity={0.7} radius={[3,3,0,0]} />
                  <Bar yAxisId="sharpe" dataKey="oos_sharpe" name="OOS Sharpe" fill="#10b981" opacity={0.7} radius={[3,3,0,0]} />
                  <Line yAxisId="ret" type="monotone" dataKey="oos_return" name="OOS Ret %" stroke="#f59e0b" strokeWidth={1.5} dot={{ r: 3 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Fold details table */}
          <div className="card overflow-x-auto">
            <p className="section-title">Fold Details</p>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Fold</th>
                  <th className="hidden sm:table-cell">Train</th>
                  <th className="hidden sm:table-cell">Test</th>
                  <th>IS Sharpe</th>
                  <th>OOS Sharpe</th>
                  <th>OOS Ret</th>
                  <th className="hidden md:table-cell">OOS MDD</th>
                  <th className="hidden lg:table-cell">Best Params</th>
                </tr>
              </thead>
              <tbody>
                {result.folds.map((f, i) => (
                  <tr key={i}>
                    <td>{f.fold_index + 1}</td>
                    <td className="hidden sm:table-cell text-gray-500 text-xs">
                      {f.train_start.slice(0,10)} → {f.train_end.slice(0,10)}
                    </td>
                    <td className="hidden sm:table-cell text-gray-500 text-xs">
                      {f.test_start.slice(0,10)} → {f.test_end.slice(0,10)}
                    </td>
                    <td className={(f.is_sharpe  ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}>{two(f.is_sharpe)}</td>
                    <td className={(f.oos_sharpe ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}>{two(f.oos_sharpe)}</td>
                    <td className={(f.oos_return ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                      {f.oos_return != null ? `${(f.oos_return * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="hidden md:table-cell text-red-400">
                      {f.oos_mdd != null ? `${(f.oos_mdd * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="hidden lg:table-cell text-gray-400 text-xs">
                      {Object.entries(f.best_params).map(([k, v]) => `${k}=${v}`).join(', ')}
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
