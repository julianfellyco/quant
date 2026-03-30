// api.ts — typed fetch wrappers for all backend endpoints

import type {
  TickersResponse,
  BacktestRequest,
  BacktestResponse,
  PairsRequest,
  PairsResponse,
  StressRequest,
  StressResponse,
  WalkForwardRequest,
  WalkForwardResponse,
  UniverseRequest,
  UniverseResponse,
  PairsScanRequest,
  PairsScanResponse,
} from './types'

const BASE = '/api'

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${text}`)
  }
  return res.json() as Promise<T>
}

export async function fetchTickers(): Promise<TickersResponse> {
  const res = await fetch(`${BASE}/tickers`)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<TickersResponse>
}

export const runBacktest = (req: BacktestRequest) =>
  post<BacktestResponse>('/backtest', req)

export const runPairs = (req: PairsRequest) =>
  post<PairsResponse>('/pairs', req)

export const runStress = (req: StressRequest) =>
  post<StressResponse>('/stress', req)

export const runWalkForward = (req: WalkForwardRequest) =>
  post<WalkForwardResponse>('/walkforward', req)

export const runUniverse = (req: UniverseRequest) =>
  post<UniverseResponse>('/universe', req)

export const scanPairs = (req: PairsScanRequest) =>
  post<PairsScanResponse>('/pairs/scan', req)
