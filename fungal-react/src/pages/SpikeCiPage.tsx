import { useEffect, useMemo, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts'

type CiIndex = {
  species: string
  spike_rate_ci: {
    t_center_s: number[]
    mean: number[]
    lo: number[]
    hi: number[]
  }
}

export default function SpikeCiPage() {
  const [data, setData] = useState<CiIndex | null>(null)

  useEffect(() => {
    fetch('/data/Schizophyllum_commune/ci/index.json')
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData(null))
  }, [])

  const chartData = useMemo(() => {
    if (!data) return []
    const t = data.spike_rate_ci.t_center_s
    const mean = data.spike_rate_ci.mean
    const lo = data.spike_rate_ci.lo
    const hi = data.spike_rate_ci.hi
    return t.map((tt, i) => ({ t: tt, mean: mean[i], lo: lo[i], hi: hi[i] }))
  }, [data])

  return (
    <div style={{ padding: 16 }}>
      <h1>Spike Rate CI</h1>
      {!data && <p>Loading…</p>}
      {data && (
        <div style={{ height: 360 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="t" label={{ value: 't_center_s', position: 'insideBottomRight', offset: -5 }} />
              <YAxis label={{ value: 'spike rate', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="mean" stroke="#1976d2" dot={false} name="Mean" />
              <Line type="monotone" dataKey="lo" stroke="#ef6c00" dot={false} name="Lower CI" />
              <Line type="monotone" dataKey="hi" stroke="#2e7d32" dot={false} name="Upper CI" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

