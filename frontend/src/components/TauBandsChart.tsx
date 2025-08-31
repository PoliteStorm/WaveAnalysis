import { useEffect, useMemo, useState } from 'react'
import Papa from 'papaparse'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from 'recharts'

type TauRow = {
  time_s: number
  tau_5_5: number
  tau_24_5: number
  tau_104: number
  tau_5_5_norm: number
  tau_24_5_norm: number
  tau_104_norm: number
}

function normalizeHeader(h: string): string {
  return h.replaceAll('.', '_').replaceAll('-', '_')
}

export function TauBandsChart({ url }: { url: string }) {
  const [rows, setRows] = useState<TauRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetch(url)
      .then((r) => r.text())
      .then((text) => {
        const parsed = Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true, comments: '#', transformHeader: normalizeHeader })
        const data: TauRow[] = []
        for (const rec of parsed.data as any[]) {
          if (!rec || typeof rec.time_s !== 'number') continue
          data.push({
            time_s: rec.time_s,
            tau_5_5: rec.tau_5_5,
            tau_24_5: rec.tau_24_5,
            tau_104: rec.tau_104,
            tau_5_5_norm: rec.tau_5_5_norm,
            tau_24_5_norm: rec.tau_24_5_norm,
            tau_104_norm: rec.tau_104_norm,
          })
        }
        setRows(data)
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false))
  }, [url])

  const series = useMemo(() => rows.map((r) => ({
    x: r.time_s,
    fast: r.tau_5_5_norm ?? r.tau_5_5,
    medium: r.tau_24_5_norm ?? r.tau_24_5,
    slow: r.tau_104_norm ?? r.tau_104,
  })), [rows])

  if (loading) return <div>Loading tau bands…</div>
  if (error) return <div>Error: {error}</div>

  return (
    <div style={{ width: '100%', height: 280 }}>
      <ResponsiveContainer>
        <LineChart data={series} margin={{ top: 10, left: 10, right: 10, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="x" type="number" label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }} />
          <YAxis label={{ value: 'Normalized Power', angle: -90, position: 'insideLeft' }} />
          <Tooltip formatter={(v, n) => [typeof v === 'number' ? v.toFixed(3) : v, n]} />
          <Legend />
          <Line type="monotone" dataKey="fast" stroke="#ef6c00" dot={false} />
          <Line type="monotone" dataKey="medium" stroke="#1976d2" dot={false} />
          <Line type="monotone" dataKey="slow" stroke="#2e7d32" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

