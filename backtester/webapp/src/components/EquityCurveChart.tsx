import {
  ResponsiveContainer, AreaChart, Area,
  XAxis, YAxis, Tooltip, CartesianGrid,
} from 'recharts'
import type { EquityCurvePoint } from '../types'

interface Props {
  data:  EquityCurvePoint[]
  label: string
}

function fmt(v: number) {
  return v >= 1_000 ? `$${(v / 1_000).toFixed(1)}k` : `$${v.toFixed(0)}`
}

export default function EquityCurveChart({ data, label }: Props) {
  if (!data.length) return null

  const slim = data.map(d => ({
    t:      d.timestamp.slice(0, 10),
    equity: Math.round(d.equity),
  }))

  // Downsample to ≤ 400 points for performance
  const step   = Math.max(1, Math.floor(slim.length / 400))
  const points = slim.filter((_, i) => i % step === 0)

  const minVal = Math.min(...points.map(p => p.equity))
  const maxVal = Math.max(...points.map(p => p.equity))
  const isUp   = points[points.length - 1].equity >= points[0].equity

  return (
    <div className="w-full">
      <p className="text-xs text-gray-500 mb-2">{label}</p>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={points} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id={`grad-${label}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor={isUp ? '#10b981' : '#ef4444'} stopOpacity={0.25} />
              <stop offset="95%" stopColor={isUp ? '#10b981' : '#ef4444'} stopOpacity={0}    />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="t"
            tick={{ fill: '#6b7280', fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[minVal * 0.995, maxVal * 1.005]}
            tickFormatter={fmt}
            tick={{ fill: '#6b7280', fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            width={52}
          />
          <Tooltip
            contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: '#9ca3af' }}
            formatter={(v: number) => [fmt(v), 'Equity']}
          />
          <Area
            type="monotone"
            dataKey="equity"
            stroke={isUp ? '#10b981' : '#ef4444'}
            strokeWidth={1.5}
            fill={`url(#grad-${label})`}
            dot={false}
            activeDot={{ r: 3, strokeWidth: 0 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
