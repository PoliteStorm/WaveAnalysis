import { useEffect, useMemo, useState } from 'react'
import Papa from 'papaparse'
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts'

type SpikeRow = { t_s: number }

export function SpikeRaster({ url }: { url: string }) {
  const [rows, setRows] = useState<SpikeRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetch(url)
      .then((r) => r.text())
      .then((text) => {
        const parsed = Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true, comments: '#', })
        const data: SpikeRow[] = []
        for (const rec of parsed.data as any[]) {
          if (rec && typeof rec.t_s === 'number') data.push({ t_s: rec.t_s })
        }
        setRows(data)
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false))
  }, [url])

  const points = useMemo(() => rows.map((r) => ({ x: r.t_s, y: 1 })), [rows])

  if (loading) return <div>Loading spikes…</div>
  if (error) return <div>Error: {error}</div>

  return (
    <div style={{ width: '100%', height: 240 }}>
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 10, bottom: 20, left: 10, right: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="x" type="number" name="time (s)" label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }} />
          <YAxis dataKey="y" type="number" hide domain={[0, 1]} />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(v, n) => [n === 'x' ? `${v} s` : v, n]} />
          <Scatter data={points} fill="#8884d8" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}

