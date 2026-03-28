import { useState, useEffect } from 'react'
import type { Tab, TickerInfo } from './types'
import { fetchTickers } from './api'
import BacktestTab     from './components/BacktestTab'
import PairsTab        from './components/PairsTab'
import StressTab       from './components/StressTab'
import WalkForwardTab  from './components/WalkForwardTab'

const TABS: { id: Tab; label: string }[] = [
  { id: 'backtest',    label: 'Backtest'      },
  { id: 'pairs',       label: 'Pairs'         },
  { id: 'stress',      label: 'Stress Test'   },
  { id: 'walkforward', label: 'Walk-Forward'  },
]

export default function App() {
  const [tab,     setTab]     = useState<Tab>('backtest')
  const [tickers, setTickers] = useState<TickerInfo[]>([])

  useEffect(() => {
    fetchTickers()
      .then(r => setTickers(r.tickers))
      .catch(console.error)
  }, [])

  return (
    <div className="min-h-screen flex flex-col">
      {/* ── Header ─────────────────────────────────────────────── */}
      <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="text-2xl select-none">📈</span>
            <span className="font-bold text-gray-100 hidden sm:block">Pharma Backtester</span>
          </div>

          {/* Tab nav */}
          <nav className="flex gap-1 overflow-x-auto">
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={tab === t.id ? 'tab-btn-active' : 'tab-btn'}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* ── Body ───────────────────────────────────────────────── */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6">
        {tab === 'backtest'    && <BacktestTab    tickers={tickers} />}
        {tab === 'pairs'       && <PairsTab />}
        {tab === 'stress'      && <StressTab      tickers={tickers} />}
        {tab === 'walkforward' && <WalkForwardTab tickers={tickers} />}
      </main>
    </div>
  )
}
