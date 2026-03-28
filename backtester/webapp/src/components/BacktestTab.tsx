import { useState } from 'react'
import type { TickerInfo, BacktestResponse } from '../types'
import { runBacktest } from '../api'
import EquityCurveChart  from './EquityCurveChart'
import MetricsGrid       from './MetricsGrid'
import EventDecompPanel  from './EventDecompPanel'
import ComparisonTable   from './ComparisonTable'

interface Props { tickers: TickerInfo[] }

const STRATEGIES = ['momentum', 'mean_reversion']

export default function BacktestTab({ tickers }: Props) {
  const [selectedTickers,  setSelectedTickers]  = useState<string[]>(['NVO', 'PFE'])
  const [selectedStrats,   setSelectedStrats]   = useState<string[]>(['momentum'])
  const [startDate,        setStartDate]        = useState('2023-01-01')
  const [endDate,          setEndDate]          = useState('2024-12-31')
  const [granularity,      setGranularity]      = useState<'daily' | 'hour' | 'minute'>('daily')
  const [initialCapital,   setInitialCapital]   = useState(100_000)
  const [sharesPerUnit,    setSharesPerUnit]    = useState(100)
  const [riskFreeRate,     setRiskFreeRate]     = useState(0.05)
  const [useEventHedge,    setUseEventHedge]    = useState(false)
  const [entryZ,           setEntryZ]           = useState(1.5)
  const [exitZ,            setExitZ]            = useState(0.3)

  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState<string | null>(null)
  const [result,   setResult]   = useState<BacktestResponse | null>(null)

  function toggleTicker(t: string) {
    setSelectedTickers(prev =>
      prev.includes(t) ? prev.filter(x => x !== t) : [...prev, t]
    )
  }

  function toggleStrat(s: string) {
    setSelectedStrats(prev =>
      prev.includes(s) ? prev.filter(x => x !== s) : [...prev, s]
    )
  }

  async function submit() {
    if (!selectedTickers.length || !selectedStrats.length) return
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await runBacktest({
        tickers:         selectedTickers,
        start_date:      startDate,
        end_date:        endDate,
        strategies:      selectedStrats,
        granularity,
        initial_capital: initialCapital,
        shares_per_unit: sharesPerUnit,
        risk_free_rate:  riskFreeRate,
        use_event_hedge: useEventHedge,
        entry_z:         entryZ,
        exit_z:          exitZ,
      })
      setResult(res)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const knownTickers = tickers.length
    ? tickers.map(t => t.ticker)
    : ['NVO', 'PFE']

  return (
    <div className="space-y-6">
      {/* ── Params card ──────────────────────────────────────── */}
      <div className="card space-y-5">
        <p className="section-title">Parameters</p>

        {/* Tickers */}
        <div>
          <p className="label">Tickers</p>
          <div className="flex flex-wrap gap-2">
            {knownTickers.map(t => (
              <button
                key={t}
                onClick={() => toggleTicker(t)}
                className={`px-3 py-1 rounded-full text-xs font-semibold border transition-colors
                  ${selectedTickers.includes(t)
                    ? 'bg-brand-600 border-brand-500 text-white'
                    : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-500'
                  }`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        {/* Strategies */}
        <div>
          <p className="label">Strategies</p>
          <div className="flex flex-wrap gap-2">
            {STRATEGIES.map(s => (
              <button
                key={s}
                onClick={() => toggleStrat(s)}
                className={`px-3 py-1 rounded-full text-xs font-semibold border transition-colors capitalize
                  ${selectedStrats.includes(s)
                    ? 'bg-brand-600 border-brand-500 text-white'
                    : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-500'
                  }`}
              >
                {s.replace('_', ' ')}
              </button>
            ))}
          </div>
        </div>

        {/* Date + granularity row */}
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
            <label className="label">Initial Capital</label>
            <input type="number" className="input" value={initialCapital}
              onChange={e => setInitialCapital(Number(e.target.value))} min={1000} step={10000} />
          </div>
        </div>

        {/* Numeric params row */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div>
            <label className="label">Shares / Unit</label>
            <input type="number" className="input" value={sharesPerUnit}
              onChange={e => setSharesPerUnit(Number(e.target.value))} min={1} />
          </div>
          <div>
            <label className="label">Risk-Free Rate</label>
            <input type="number" className="input" value={riskFreeRate}
              onChange={e => setRiskFreeRate(Number(e.target.value))} step={0.01} min={0} max={0.2} />
          </div>
          <div>
            <label className="label">Entry Z</label>
            <input type="number" className="input" value={entryZ}
              onChange={e => setEntryZ(Number(e.target.value))} step={0.1} min={0.5} max={4} />
          </div>
          <div>
            <label className="label">Exit Z</label>
            <input type="number" className="input" value={exitZ}
              onChange={e => setExitZ(Number(e.target.value))} step={0.1} min={0} max={2} />
          </div>
        </div>

        {/* Event hedge toggle + run button */}
        <div className="flex items-center justify-between flex-wrap gap-3">
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <div
              onClick={() => setUseEventHedge(v => !v)}
              className={`w-10 h-5 rounded-full relative transition-colors ${useEventHedge ? 'bg-brand-600' : 'bg-gray-700'}`}
            >
              <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform
                ${useEventHedge ? 'translate-x-5' : ''}`} />
            </div>
            <span className="text-sm text-gray-300">Event Hedge</span>
          </label>

          <button className="btn-primary" onClick={submit} disabled={loading}>
            {loading
              ? <><span className="spinner" /> Running…</>
              : 'Run Backtest'}
          </button>
        </div>
      </div>

      {/* ── Error ───────────────────────────────────────────── */}
      {error && <div className="error-banner">{error}</div>}

      {/* ── Results ─────────────────────────────────────────── */}
      {result && (
        <div className="space-y-6">
          {result.results.map((r, i) => (
            <div key={i} className="space-y-3">
              <h2 className="text-sm font-semibold text-gray-300">
                {r.metrics.ticker} — <span className="capitalize">{r.metrics.strategy.replace('_', ' ')}</span>
              </h2>
              <MetricsGrid m={r.metrics} />
              <EquityCurveChart data={r.equity_curve} label="Equity Curve" />
              <EventDecompPanel result={r} />
            </div>
          ))}

          {result.comparison_table.length > 1 && (
            <ComparisonTable rows={result.comparison_table} />
          )}
        </div>
      )}
    </div>
  )
}
