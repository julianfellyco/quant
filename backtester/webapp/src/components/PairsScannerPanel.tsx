import { useState } from 'react'
import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis,
  Tooltip, CartesianGrid, Cell,
} from 'recharts'
import type { PairsScanResponse, PairResult } from '../types'
import { scanPairs } from '../api'

const UNIVERSES = [
  { id: 'pharma',  label: 'Pharma',  desc: '10 large-cap pharma tickers' },
  { id: 'tech',    label: 'Tech',    desc: '10 mega-cap tech tickers'     },
  { id: 'energy',  label: 'Energy',  desc: '10 energy sector tickers'     },
  { id: 'custom',  label: 'Custom',  desc: 'Enter your own tickers'       },
]

export default function PairsScannerPanel() {
  const [universe,       setUniverse]       = useState<string>('pharma')
  const [customInput,    setCustomInput]    = useState('PFE, NVO, LLY, JNJ, MRK, ABBV')
  const [startDate,      setStartDate]      = useState('2022-01-01')
  const [endDate,        setEndDate]        = useState('2024-12-31')
  const [lookback,       setLookback]       = useState(252)
  const [minPvalue,      setMinPvalue]      = useState(0.05)
  const [maxHalfLife,    setMaxHalfLife]    = useState(60)
  const [loading,        setLoading]        = useState(false)
  const [error,          setError]          = useState<string | null>(null)
  const [result,         setResult]         = useState<PairsScanResponse | null>(null)

  function parseCustom() {
    return customInput.split(/[,\s]+/).map(t => t.trim().toUpperCase()).filter(Boolean)
  }

  async function submit() {
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await scanPairs({
        universe,
        custom_tickers: universe === 'custom' ? parseCustom() : [],
        start_date: startDate,
        end_date: endDate,
        lookback_days: lookback,
        min_coint_pvalue: minPvalue,
        max_half_life: maxHalfLife,
        min_half_life: 5,
        max_hurst: 0.5,
      })
      setResult(res)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const scatterData = result?.pairs.map(p => ({
    x: p.half_life,
    y: p.hurst_exponent,
    label: `${p.ticker_a}/${p.ticker_b}`,
    tradeable: p.is_tradeable,
    pvalue: p.coint_pvalue,
  })) ?? []

  return (
    <div className="space-y-6">
      <div className="card space-y-5">
        <p className="section-title">Pairs Cointegration Scanner</p>

        {/* Universe selector */}
        <div>
          <p className="label">Universe</p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {UNIVERSES.map(u => (
              <button key={u.id} onClick={() => setUniverse(u.id)}
                className={`rounded-lg border px-3 py-2 text-left transition-colors
                  ${universe === u.id
                    ? 'bg-brand-600 border-brand-500 text-white'
                    : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600'}`}>
                <p className="text-xs font-semibold">{u.label}</p>
                <p className="text-xs opacity-60 mt-0.5">{u.desc}</p>
              </button>
            ))}
          </div>
        </div>

        {universe === 'custom' && (
          <div>
            <label className="label">Tickers <span className="text-gray-600 font-normal">(comma separated)</span></label>
            <input className="input" value={customInput} onChange={e => setCustomInput(e.target.value)}
              placeholder="PFE, NVO, LLY, JNJ" />
          </div>
        )}

        {/* Params */}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          <div>
            <label className="label">Start Date</label>
            <input type="date" className="input" value={startDate} onChange={e => setStartDate(e.target.value)} />
          </div>
          <div>
            <label className="label">End Date</label>
            <input type="date" className="input" value={endDate} onChange={e => setEndDate(e.target.value)} />
          </div>
          <div>
            <label className="label">Lookback (days)</label>
            <input type="number" className="input" value={lookback} min={60} max={756}
              onChange={e => setLookback(Number(e.target.value))} />
          </div>
          <div>
            <label className="label">Max p-value</label>
            <input type="number" className="input" value={minPvalue} min={0.01} max={0.20} step={0.01}
              onChange={e => setMinPvalue(Number(e.target.value))} />
          </div>
          <div>
            <label className="label">Max Half-Life (days)</label>
            <input type="number" className="input" value={maxHalfLife} min={5} max={252}
              onChange={e => setMaxHalfLife(Number(e.target.value))} />
          </div>
        </div>

        <div className="flex justify-end">
          <button className="btn-primary" onClick={submit} disabled={loading}>
            {loading ? <><span className="spinner" /> Scanning pairs…</> : 'Scan for Pairs'}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <div className="space-y-4">
          {/* Scan summary */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {[
              { label: 'Universe Size',     value: result.scan_metadata.universe_size },
              { label: 'Pairs Tested',      value: result.scan_metadata.pairs_tested },
              { label: 'Cointegrated',      value: result.scan_metadata.pairs_cointegrated },
              { label: 'Tradeable',         value: result.scan_metadata.pairs_tradeable },
            ].map(m => (
              <div key={m.label} className="metric-box">
                <span className="metric-label">{m.label}</span>
                <span className="metric-value">{m.value}</span>
              </div>
            ))}
          </div>

          {/* Scatter: half-life vs hurst */}
          {scatterData.length > 0 && (
            <div className="card">
              <p className="section-title">Half-Life vs Hurst — Tradeable Zone (H &lt; 0.5, HL 5–60d)</p>
              <ResponsiveContainer width="100%" height={280}>
                <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="x" name="Half-Life (days)"
                    tick={{ fill: '#6b7280', fontSize: 10 }} label={{ value: 'Half-Life (days)', fill: '#6b7280', fontSize: 10, position: 'insideBottom', offset: -4 }} />
                  <YAxis dataKey="y" name="Hurst"
                    tick={{ fill: '#6b7280', fontSize: 10 }} label={{ value: 'Hurst', fill: '#6b7280', fontSize: 10, angle: -90, position: 'insideLeft' }} />
                  <Tooltip
                    contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8, fontSize: 11 }}
                    formatter={(v: number, name: string) => [v.toFixed(3), name]}
                    labelFormatter={(_, payload) => payload?.[0]?.payload?.label ?? ''}
                  />
                  <Scatter data={scatterData}>
                    {scatterData.map((d, i) => (
                      <Cell key={i} fill={d.tradeable ? '#10b981' : '#6b7280'} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
              <p className="text-xs text-gray-600 mt-1">Green = tradeable (H &lt; 0.5 and 5d ≤ half-life ≤ 60d)</p>
            </div>
          )}

          {/* Results table */}
          {result.pairs.length > 0 && (
            <div className="card overflow-x-auto">
              <p className="section-title">All Cointegrated Pairs</p>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Pair</th>
                    <th>p-value</th>
                    <th>Half-Life</th>
                    <th>Hurst</th>
                    <th className="hidden md:table-cell">Hedge Ratio</th>
                    <th className="hidden md:table-cell">Spread Vol</th>
                    <th className="hidden md:table-cell">Corr</th>
                    <th>Tradeable</th>
                  </tr>
                </thead>
                <tbody>
                  {result.pairs.map((p: PairResult, i: number) => (
                    <tr key={i}>
                      <td className="font-semibold text-brand-400">{p.ticker_a}/{p.ticker_b}</td>
                      <td>{p.coint_pvalue.toFixed(4)}</td>
                      <td>{p.half_life.toFixed(1)}d</td>
                      <td className={p.hurst_exponent < 0.5 ? 'text-emerald-400' : 'text-gray-400'}>{p.hurst_exponent.toFixed(3)}</td>
                      <td className="hidden md:table-cell">{p.hedge_ratio.toFixed(3)}</td>
                      <td className="hidden md:table-cell">{(p.spread_volatility * 100).toFixed(1)}%</td>
                      <td className="hidden md:table-cell">{p.correlation.toFixed(3)}</td>
                      <td>{p.is_tradeable ? <span className="text-emerald-400 font-semibold">✓</span> : <span className="text-gray-600">—</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {result.pairs.length === 0 && (
            <div className="card text-center text-gray-500 py-8">
              No cointegrated pairs found with the current filters. Try relaxing the p-value threshold or extending the date range.
            </div>
          )}
        </div>
      )}
    </div>
  )
}
